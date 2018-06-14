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



def fc(inputs, num_outputs, activation=tf.nn.relu, name=None):
    return tf.layers.dense(inputs, num_outputs, activation=activation,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
        name=name)

def conv2d(inputs, filters, kernel_size, strides=(1,1), padding='same', activation=tf.nn.relu,
    trainable=True, name=None):
    return tf.layers.conv2d(inputs, filters, kernel_size, strides=strides, padding=padding,
        activation=activation, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
        trainable=trainable, name=name)


def train(sess, nbatch, data_train, data_test):
    hidden_size = 128
    num_steps = 6
    answer_size = 10

    def get_cell():
        cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
        #cell = tf.contrib.rnn.LayerNormBasicLSTMCell(hidden_size)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.8)
        return cell

    cell = tf.contrib.rnn.MultiRNNCell([get_cell()])
    # TODO try dropout

    b_images = tf.placeholder(tf.float32, shape=[nbatch,75,75,3])
    b_questions = tf.placeholder(tf.float32, shape=[nbatch,11])
    b_answers = tf.placeholder(tf.int32, shape=[nbatch])

    resized_images = tf.image.resize_images(b_images, [64, 64])
    conv_1 = conv2d(resized_images, 24, 7, strides=(2,2)) # 32
    conv_2 = conv2d(conv_1, 24, 5, strides=(2,2)) # 16
    conv_3 = conv2d(conv_2, 24, 3, strides=(2,2)) # 8
    #conv_3 = tf.layers.flatten(conv_3)

    #fc_1 = fc(tf.concat([conv_3, b_questions], axis=1), 256)
    #fc_2 = fc(fc_1, 256)

    #image_features = # TODO
    #attention_map = 

    state = cell.zero_state(nbatch, tf.float32)
    cell_output = tf.zeros([nbatch,hidden_size])
    att = tf.ones([nbatch,8,8,1])
    outputs = []
    with tf.variable_scope('RNN'):
        for time_step in range(num_steps):
            if time_step > 0:
                tf.get_variable_scope().reuse_variables()

            fc_1 = fc(tf.concat([cell_output, b_questions], axis=1), 128, name='fc_1')

            fc_2 = fc(fc_1, 128, name='fc_2')
            fc_2 = fc(fc_2, 64, name='fc_22')
            att = tf.tile(tf.reshape(fc_2, [-1,8,8,1]), [1,1,1,24]) + att
            conv_att = tf.multiply(conv_3, att)
            conv_att = tf.reduce_sum(conv_att, axis=[1,2])
            conv_att = tf.layers.flatten(conv_att)

            fc_2_2 = fc(fc_1, 128, name='fc_2_2')
            fc_2_2 = fc(fc_2_2, 64, name='fc_22_2')
            att_2 = tf.tile(tf.reshape(fc_2, [-1,8,8,1]), [1,1,1,24])
            conv_att_2 = tf.multiply(conv_3, att_2)
            conv_att_2 = tf.reduce_sum(conv_att_2, axis=[1,2])
            conv_att_2 = tf.layers.flatten(conv_att_2)

            conv_att = tf.concat([conv_att, conv_att_2, b_questions], axis=1)

            fc_3 = fc(conv_att, 128, name='fc_3')
            fc_3_2 = fc(fc_3, 128, name='fc_3_2', activation=None)

            x_t = fc_3
            cell_output, state = cell(x_t, state)
            output_t = cell_output + x_t
            outputs.append(output_t)

    output = tf.reshape(tf.concat(outputs, axis=1), [-1, hidden_size])
    print(output)

    output = fc(output, 128)
    output = fc(output, 128)
    output = tf.layers.dropout(output, 0.2)
    output = fc(output, answer_size, activation=None)

    output = tf.reshape(output, [nbatch,num_steps,answer_size])

    losses = []
    #loss_coefs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.8, 0.9, 1.0]
    #loss_coefs = [0.2, 0.3, 0.4, 0.5, 0.6, 1.0]
    #loss_coefs = [0.1, 0.1, 0.1, 0.3, 0.4, 1.0]
    loss_coefs = [1.0, 0.8, 0.4, 0.3, 0.3, 0.5]
    #loss_coefs = [0.05, 0.1, 0.3, 1.0]
    for time_step in range(num_steps):
        logit = output[:,time_step,:]
        losses.append(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=b_answers, logits=output[:,time_step,:]))
        losses[-1] = loss_coefs[time_step] * losses[-1]
    loss = tf.reduce_sum(losses)
    final_loss = tf.reduce_mean(losses[-1])
    predictions = tf.argmax(output[:,-1,:], axis=1)

    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss,tvars), 10)
    optimizer = tf.train.AdamOptimizer(0.001)
    train_op = optimizer.apply_gradients(zip(grads, tvars))

    sess.run(tf.global_variables_initializer())

    smooth_loss = 0
    smooth_train_acc = 0
    for gs in range(100000):
        sample_images, sample_questions, sample_answers = sample_batch(nbatch, data_train)
        #print(np.min(sample_answers), np.max(sample_answers))

        sess.run(train_op, feed_dict={
            b_images: sample_images,
            b_questions: sample_questions,
            b_answers: sample_answers
        })

        train_loss, preds = sess.run([final_loss, predictions], feed_dict={
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

        
        if gs % 500 == 0:
            test_loss = 0
            test_correct = 0
            num_batches = len(data_test) // nbatch
            for i in range(num_batches):
                sample_images, sample_questions, sample_answers = sample_batch(nbatch, data_test, i)
                batch_loss, batch_preds = sess.run([final_loss, predictions], feed_dict={
                    b_images: sample_images,
                    b_questions: sample_questions,
                    b_answers: sample_answers
                })
                #print(batch_preds)
                test_loss += batch_loss
                test_correct += np.sum([1 if p==ans else 0 for p,ans in zip(batch_preds,sample_answers)])
            test_loss /= num_batches
            test_acc = test_correct / len(data_test)
            print('it {}: test_loss={}, test_accuracy={}'.format(gs, test_loss, test_acc))

        if gs % 100 == 0:
            print('it {}: smooth_loss={}, train_loss={}, smooth_train_acc={}, train_acc={}'.format(
                gs, smooth_loss, train_loss, smooth_train_acc, train_acc))








def main():
    rel_train, rel_test, norel_train, norel_test = load_data()
    print(len(rel_test))
    print(len(rel_train))

    with tf.Session() as sess:
        #train(sess, 100, norel_train, norel_test)
        train(sess, 100, rel_train, rel_test)

main()
