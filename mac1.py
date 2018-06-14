# DOESN'T REALLY WORK

import tensorflow as tf
import numpy as np
import pickle
import os


def load_data():
    print('loading data...')
    dirs = './data'
    filename = os.path.join(dirs,'sort-of-clevr.pickle')
    with open(filename, 'rb') as f:
        train_datasets, test_datasets = pickle.load(f)
    rel_train = []
    rel_test = []
    norel_train = []
    norel_test = []
    print('processing data...')

    for img, relations, norelations in train_datasets:
        for qst,ans in zip(relations[0], relations[1]):
            rel_train.append((img,qst,ans))
        for qst,ans in zip(norelations[0], norelations[1]):
            norel_train.append((img,qst,ans))

    for img, relations, norelations in test_datasets:
        for qst,ans in zip(relations[0], relations[1]):
            rel_test.append((img,qst,ans))
        for qst,ans in zip(norelations[0], norelations[1]):
            norel_test.append((img,qst,ans))

    return (rel_train, rel_test, norel_train, norel_test)

def sample_batch(nbatch, data, start=None):
    images = []
    questions = []
    answers = []
    if start:
        image_idxs = np.arange(start*nbatch, (start+1)*nbatch)
    image_idxs = np.random.choice(len(data), nbatch)
    for i in image_idxs:
        images.append(data[i][0])
        questions.append(data[i][1])
        answers.append(data[i][2])
    return np.array(images), np.array(questions), np.array(answers)



#def fc(inputs, num_outputs, activation=tf.nn.relu, name=None):
def fc(inputs, num_outputs, activation=tf.nn.elu, name=None):
    return tf.layers.dense(inputs, num_outputs, activation=activation,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
        name=name)

def conv2d(inputs, filters, kernel_size, strides=(1,1), padding='same', activation=tf.nn.relu,
    trainable=True, name=None):
    return tf.layers.conv2d(inputs, filters, kernel_size, strides=strides, padding=padding,
        activation=activation, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
        trainable=trainable, name=name)

def repeat(x, dim, n):
    # repeat dim times along axis 0
    return tf.reshape(tf.tile(tf.reshape(x, [-1,dim,1]), [1,n,1]), [-1,dim])


def control(control_in, question, dim, reuse=None):
    with tf.variable_scope('control', reuse=reuse):
        cq_i = fc(tf.concat([control_in, question], axis=1), dim, name='fc_1')
        cq_i = fc(cq_i, dim, activation=tf.nn.softmax, name='fc_2') # TODO use another activation?

        return cq_i


def read(control_in, mem_prev, k, dim, n_objs, reuse=None):
    with tf.variable_scope('read', reuse=reuse):
        fc_1 = fc(mem_prev, dim, activation=None, name='fc_1')
        fc_1 = repeat(fc_1, dim, n_objs)
        fc_2 = fc(k, dim, activation=None, name='fc_2')
        I = tf.multiply(fc_1, fc_2)
        
        I_prime = fc(tf.concat([I, k], axis=1), dim)

        c_i = tf.multiply(repeat(control_in, dim, n_objs), I_prime)
        ra = fc(c_i, dim, activation=None, name='fc_3')
        rv = tf.nn.softmax(ra)

        r_i = tf.multiply(rv, k)
        r_i = tf.reduce_sum(tf.reshape(r_i, [-1,n_objs,dim]), axis=[1])

        return r_i


def write(read_in, mem_prev, control_in, dim, reuse=None):
    with tf.variable_scope('write', reuse=reuse):
        m_i_info = fc(tf.concat([read_in, mem_prev], axis=1), dim, name='fc_1')

        return m_i_info


def mac_cell(control_in, question, mem_prev, k, dim, n_objs, reuse=None):
    mem_prev = tf.nn.dropout(mem_prev, 0.85)
    c_i = control(control_in, question, dim, reuse=reuse)
    c_i = tf.nn.dropout(c_i, 0.85)
    r_i = read(c_i, mem_prev, k, dim, n_objs, reuse=reuse)
    r_i = tf.nn.dropout(r_i, 0.9)
    m_i = write(r_i, mem_prev, c_i, dim, reuse=reuse)

    return c_i, m_i


def train(sess, nbatch, data_train, data_test):
    dim = 128
    n_objs = 8*8
    answer_size = 10
    num_steps = 6

    b_images = tf.placeholder(tf.float32, shape=[nbatch,75,75,3])
    b_questions = tf.placeholder(tf.float32, shape=[nbatch,11])
    b_answers = tf.placeholder(tf.int32, shape=[nbatch])

    resized_images = tf.image.resize_images(b_images, [64, 64])
    conv_1 = conv2d(resized_images, 24, 7, strides=(2,2)) # 32
    conv_2 = conv2d(conv_1, 24, 5, strides=(2,2)) # 16
    conv_3 = conv2d(conv_2, dim, 3, strides=(2,2)) # 8
    k = tf.reshape(conv_3, [nbatch*n_objs,dim])

    # initial values
    c_i = tf.tile(tf.Variable(tf.zeros([1,dim]), dtype=tf.float32), [nbatch, 1])
    m_i = tf.tile(tf.Variable(tf.zeros([1,dim]), dtype=tf.float32), [nbatch, 1])
    for i in range(num_steps):
        reuse = False
        if i > 0:
            reuse = True

        question = fc(b_questions, 11, activation=tf.nn.tanh, name='qfc_{}'.format(i))

        c_i, m_i = mac_cell(c_i, question, m_i, k, dim, n_objs, reuse)

    output = fc(tf.concat([b_questions, m_i], axis=1), 64)
    output = tf.nn.dropout(output, 0.85)
    output = fc(output, answer_size, activation=tf.nn.softmax)


    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=b_answers, logits=output)
    loss = tf.reduce_mean(loss)
    predictions = tf.argmax(output, axis=1)

    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss,tvars), 7)
    optimizer = tf.train.AdamOptimizer(0.0001)
    train_op = optimizer.apply_gradients(zip(grads, tvars))

    sess.run(tf.global_variables_initializer())

    smooth_loss = 0
    smooth_train_acc = 0
    for gs in range(1000000):
        sample_images, sample_questions, sample_answers = sample_batch(nbatch, data_train)
        #print(np.min(sample_answers), np.max(sample_answers))

        sess.run(train_op, feed_dict={
            b_images: sample_images,
            b_questions: sample_questions,
            b_answers: sample_answers
        })

        train_loss, preds = sess.run([loss, predictions], feed_dict={
            b_images: sample_images,
            b_questions: sample_questions,
            b_answers: sample_answers
        })

        train_correct = np.sum([1 if p==ans else 0 for p,ans in zip(preds,sample_answers)])
        train_acc = train_correct / nbatch

        if smooth_loss == 0:
            smooth_loss = train_loss
            smooth_train_acc = train_acc
        else:
            smooth_loss = 0.99*smooth_loss + 0.01*train_loss
            smooth_train_acc = 0.99*smooth_train_acc + 0.01*train_acc

        
        if gs % 2000 == 0:
            test_loss = 0
            test_correct = 0
            num_batches = len(data_test) // nbatch
            for i in range(num_batches):
                sample_images, sample_questions, sample_answers = sample_batch(nbatch, data_test, i)
                batch_loss, batch_preds = sess.run([loss, predictions], feed_dict={
                    b_images: sample_images,
                    b_questions: sample_questions,
                    b_answers: sample_answers
                })
                #print(batch_preds)
                test_loss += batch_loss
                test_correct += np.sum([1 if p==ans else 0 for p,ans in zip(batch_preds,sample_answers)])
            test_loss /= num_batches
            test_acc = test_correct / len(data_test)
            print(gs, test_loss, test_acc)
            print('it {}: test_loss={:.4f}, test_accuracy={:.4f}'.format(gs, test_loss, test_acc))


        if gs % 100 == 0:
            print('it {}: smooth_loss={:.4f}, train_loss={:.4f}, smooth_train_acc={:.4f}, train_acc={:.4f}'.format(
                gs, smooth_loss, train_loss, smooth_train_acc, train_acc))








def main():
    rel_train, rel_test, norel_train, norel_test = load_data()
    print(len(rel_test))
    print(len(rel_train))

    with tf.Session() as sess:
        #train(sess, 100, norel_train, norel_test)
        train(sess, 100, rel_train, rel_test)

main()
