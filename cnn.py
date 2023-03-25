import tensorflow as tf
import numpy as np
import pandas as pd


training_dataset = pd.read_csv('data/image_training_dataset_2.csv', sep=',')
training_dataset.drop(['line_id'], axis=1, inplace=True)
training_dataset = training_dataset.reindex(np.random.permutation(training_dataset.index))


#test_dataset = pd.read_csv('data/image_test_dataset_2.csv', sep=',')
#test_dataset.drop(['line_id'], axis=1, inplace=True)
#test_dataset = test_dataset.reindex(np.random.permutation(test_dataset.index))

grouped_data = pd.Series(training_dataset.groupby('writer_id').size())
authors_id = {}
k = 1
for ind in grouped_data.index:
    authors_id[ind] = k
    k += 1
#training_dataset = training_dataset.iloc[:20, :]
print(authors_id)

def output_vector(id, nmax):
    vec = np.zeros((nmax))#, dtype=np.uint8)
    vec[id-1] = 1
    return vec

def make_Y(ds):

    labels = []
    for wr in ds.loc[:, ['writer_id']].values:
        #print(wr[0], '\t', output_vector(authors_id[wr[0]], len(authors_id)))
        labels.append(output_vector(authors_id[wr[0]], len(authors_id)))

    return np.array(labels)#, dtype=np.uint8)

data_Y = make_Y(training_dataset)
data_X = training_dataset.iloc[:, 2:].values

#testdata_Y = make_Y(test_dataset)
#testdata_X = test_dataset.iloc[:, 2:].values


tf.logging.set_verbosity(tf.logging.INFO)

epoch = 10
learning_rate = 1e-3
batch_size = 32
n_output = 14
n_attr = 512
b = 0.01
w = 512
h = 32

def conv2d(x, wghts, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, wghts, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


weights = {
    'wc1': tf.get_variable('W1', shape=(3, 3, 1, 64), initializer=tf.contrib.layers.variance_scaling_initializer(),
                           trainable=True),
    'wc2': tf.get_variable('W2', shape=(3, 3, 64, 64), initializer=tf.contrib.layers.variance_scaling_initializer(),
                           trainable=True),
    'wc3': tf.get_variable('W3', shape=(3, 3, 64, 128), initializer=tf.contrib.layers.variance_scaling_initializer(),
                           trainable=True),
    'wc4': tf.get_variable('W4', shape=(3, 3, 128, 128), initializer=tf.contrib.layers.variance_scaling_initializer(),
                           trainable=True),
    'wc5': tf.get_variable('W5', shape=(3, 3, 128, 256), initializer=tf.contrib.layers.variance_scaling_initializer(),
                           trainable=True),


    'wd1_1': tf.get_variable('W1_1', shape=(1 * 16 * 256, n_attr),
                             initializer=tf.contrib.layers.variance_scaling_initializer(), trainable=True),
    #'wd1_2': tf.get_variable('W1_2', shape=(3 * 2 * 256, 256),
     #                        initializer=tf.contrib.layers.variance_scaling_initializer(), trainable=True),
    #'wd1_3': tf.get_variable('W1_3', shape=(3 * 2 * 256, 256),
     #                        initializer=tf.contrib.layers.variance_scaling_initializer(), trainable=True),

    'wd2_1': tf.get_variable('W2_1', shape=(n_attr, n_output), initializer=tf.contrib.layers.variance_scaling_initializer(),
                             trainable=True),
    #'wd2_2': tf.get_variable('W2_2', shape=(256, 9), initializer=tf.contrib.layers.variance_scaling_initializer(),
    #                         trainable=True),
    #'wd2_3': tf.get_variable('W2_3', shape=(256, 1), initializer=tf.contrib.layers.variance_scaling_initializer(),
    #                         trainable=True)

}
biases = {
    'bc1': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.variance_scaling_initializer(),
                           trainable=True),
    'bc2': tf.get_variable('B2', shape=(64), initializer=tf.contrib.layers.variance_scaling_initializer(),
                           trainable=True),
    'bc3': tf.get_variable('B3', shape=(128), initializer=tf.contrib.layers.variance_scaling_initializer(),
                           trainable=True),
    'bc4': tf.get_variable('B4', shape=(128), initializer=tf.contrib.layers.variance_scaling_initializer(),
                           trainable=True),
    'bc5': tf.get_variable('B5', shape=(256), initializer=tf.contrib.layers.variance_scaling_initializer(),
                           trainable=True),

    'bd1_1': tf.get_variable('B1_1', shape=(n_attr), initializer=tf.contrib.layers.variance_scaling_initializer(),
                             trainable=True),
    'bd2_1': tf.get_variable('B2_1', shape=(n_output), initializer=tf.contrib.layers.variance_scaling_initializer(),
                             trainable=True),
}


def conv_net(x, weights, biases):

    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    pool1 = maxpool2d(conv1, k=2)
    print("conv1", pool1.get_shape().as_list())

    conv2 = conv2d(pool1, weights['wc2'], biases['bc2'])
    pool2 = maxpool2d(conv2, k=2)
    print("conv2", pool2.get_shape().as_list())

    conv3 = conv2d(pool2, weights['wc3'], biases['bc3'])
    pool3 = maxpool2d(conv3, k=2)
    print("conv3", pool3.get_shape().as_list())

    conv4 = conv2d(pool3, weights['wc4'], biases['bc4'])
    pool4 = maxpool2d(conv4, k=2)
    print("conv4", pool4.get_shape().as_list())

    conv5 = conv2d(pool4, weights['wc5'], biases['bc5'])
    pool5 = maxpool2d(conv5, k=2)
    print("pool5", pool5.get_shape().as_list())

    '''conv6 = conv2d(pool5, weights['wc6'], biases['bc6'])
    pool6 = maxpool2d(conv6, k=2)
    print("conv6", pool6.get_shape().as_list())'''

    pool5_flat = tf.reshape(pool5, [-1, pool5.shape[1] * pool5.shape[2] * pool5.shape[3]])
    #fc1_1 = tf.reshape(pool5, [-1, weights['wd1_1'].get_shape().as_list()[0]])
    fc1_11 = tf.add(tf.matmul(pool5_flat, weights['wd1_1']), biases['bd1_1'])
    fc1_111 = tf.nn.relu(fc1_11)
    print("fc1_1", fc1_111.get_shape().as_list())

    fc2_1 = tf.add(tf.matmul(fc1_111, weights['wd2_1']), biases['bd2_1'])
    # fc2_1 = tf.nn.relu(fc2_1)
    print("fc2_1", fc2_1.get_shape().as_list())

    return fc2_1

'''print(authors_id)
print(training_dataset.head(10))
print(data_Y[:10])
print(data_X[:10])
input = tf.reshape(data_X[0], [-1, 50, 800, 1])
print(input.shape)
input2 = tf.reshape(data_X[:10], [-1, 50, 800, 1])
print(input2.shape)
'''
'''def model(data_X, labels):
    # 1st convolutional layer

    input = tf.reshape(data_X, [-1, 50, 800, 1])
    conv1 = tf.layers.conv2d(inputs=input, filters=64, kernel_size=[5,5], padding="same", activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)

    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5,5], padding="same", activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)

    conv3 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    conv4 = tf.layers.conv2d(inputs=pool3, filters=128, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

    conv5 = tf.layers.conv2d(inputs=pool4, filters=256, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2)

    pool5_flat = tf.reshape(pool5, [-1, pool5.shape[1]*pool5.shape[2]*pool5.shape[3]])

    dense = tf.layers.dense(inputs=pool5_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.2)

    logits = tf.layers.dense(inputs=dropout, units=n_output)
    return logits

logits = model(data_X, data_Y)
predictions = { "author": tf.argmax(input=logits, axis=1), "probabilities": tf.nn.softmax(logits, name='softmax_tensor')}
loss = tf.losses.sparse_softmax_cross_entropy(labels=data_Y, logits=logits)

optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

eval_metric_ops = {'accuracy': tf.metrics.accuracy(labels=data_Y, predictions=predictions["author"])}
'''


X = tf.placeholder("float", [None, h, w, 1])
Y = tf.placeholder("float", [None, n_output])

logits = conv_net(X, weights, biases)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
#regularizers = tf.nn.l2_loss(weights['wd1_1']) + tf.nn.l2_loss(weights['wd2_1'])
#loss = loss + b * regularizers
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=tf.train.get_global_step())

prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy= tf.reduce_mean(tf.cast(prediction, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for j in range(epoch):
        print("Epoch " + str(j))

        for i in range(np.int(np.round(len(data_Y) / batch_size))):
            # print(i)
            batch_x = data_X[i * batch_size:(i + 1) * batch_size]
            batch_y = data_Y[i * batch_size:(i + 1) * batch_size]

            batch_x = tf.reshape(batch_x, [-1, h, w, 1])
            batch_x = batch_x.eval(session=sess)
            #print('batch_x shapes: ', batch_x.shape)
            #batch_x = batch_x.reshape((batch_size, h, w, 1))


            # Run optimization
            opt = sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y})

            if i % 10 == 0:

                logits_train, loss_train, accuracy_train = sess.run(
                    [logits, loss, accuracy],
                    feed_dict={X: batch_x, Y: batch_y})

                print(
                    "Iteration", str(i),
                    "\t| Loss =", str(loss_train),
                    "\t| Accuracy =", str(accuracy_train)
                )