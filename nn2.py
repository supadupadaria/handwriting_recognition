import tensorflow as tf
import numpy as np
import pandas as pd


training_dataset = pd.read_csv('data/scaled_training_dataset_2.csv', sep=',', usecols=['writer_id', 'length', 'area', 'height',
                                                                                     'width', 'average_direction', 'standart_deviation', 'curvature'])
training_dataset = training_dataset.reindex(np.random.permutation(training_dataset.index))


test_dataset = pd.read_csv('data/scaled_test_dataset_2.csv', sep=',', usecols=['writer_id', 'length', 'area', 'height',
                                                                                     'width', 'average_direction', 'standart_deviation', 'curvature'])

test_dataset = test_dataset.reindex(np.random.permutation(test_dataset.index))

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
data_X = training_dataset.loc[:, ['length', 'height', 'width',
                                  'average_direction', 'standart_deviation', 'curvature']].values

#print(data_Y[:3])

#print(training_dataset.iloc[:3, 0])

testdata_Y = make_Y(test_dataset)
testdata_X = test_dataset.loc[:, ['length', 'height', 'width',
                                 'average_direction', 'standart_deviation', 'curvature']].values

'''print(training_dataset.head(10))
print('\n-----------------------\n')
#print(data_X[:10])
print(len(data_X))
print('\n-----------------------\n')
#print(data_Y[:10])
print(len(data_Y))'''

learning_rate = 1e-3
n_interations = 176#1000
batch_size = 64#100
dropout = 0.5

n_input = 6
n_hidden1 = 256
n_hidden2 = 128
n_output = 10
#??

X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])
keep_prob = tf.placeholder(tf.float32)


weights = {
    'w1': tf.Variable(tf.truncated_normal([n_input, n_hidden1], stddev=0.1)),
    'w2': tf.Variable(tf.truncated_normal([n_hidden1, n_hidden2], stddev=0.1)),
    'out': tf.Variable(tf.truncated_normal([n_hidden2, n_output], stddev=0.1)),
}

biases = {
    'b1': tf.Variable(tf.constant(0.1, shape=[n_hidden1])),
    'b2': tf.Variable(tf.constant(0.1, shape=[n_hidden2])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_output]))
}

layer_1 = tf.add(tf.matmul(X, weights['w1']), biases['b1'])
layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
#Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.
#layer_drop = tf.nn.dropout(layer_3, rate=1-keep_prob) #droupout operation at last hidden layer

layer_2 = tf.nn.relu(layer_2)
output_layer = tf.matmul(layer_2, weights['out']) + biases['out']

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=output_layer))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)


correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

n_epoch = 100
for j in range(n_epoch):
    print("\nEpoch: ", j)
    for i in range(np.int(np.round(len(data_Y)/batch_size))):
        #print(i)
        batch_x = data_X[i*batch_size:(i+1)*batch_size]
        batch_y = data_Y[i*batch_size:(i+1)*batch_size]

        sess.run(train_step, feed_dict={X: batch_x, Y: batch_y})#, keep_prob: dropout})

        if i%batch_size == 0:
             minibatch_loss, minibatch_accuracy, cross_entropy_value = sess.run(
                 [cross_entropy, accuracy, cross_entropy],
                 feed_dict={X: batch_x, Y: batch_y})#, keep_prob: 1.0})

             print(
                 "Iteration", str(i),
                 "\t| Loss =", str(minibatch_loss),
                 "\t| Accuracy =", str(minibatch_accuracy)
             )
             ''' 
                "\t| Predicted =", str(cross_entropy_value),
                 "\t| Label =", str(batch_y[-1])
             )'''