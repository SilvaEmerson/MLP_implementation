from data_process import cleanData
import numpy as np
import tensorflow as tf
import math

cleanData = np.array(cleanData)

# spliting data
X_train = cleanData[:int(0.9 * len(cleanData)), :-1]
Y_train = cleanData[:int(0.9 * len(cleanData)), -1]

temp = []
for i in Y_train:
    temp.append([1, 0]) if i == 2 else temp.append([0, 1])

Y_train = np.array(temp)

X_test = cleanData[int(0.9 * len(cleanData)):, :-1]
Y_test = cleanData[int(0.9 * len(cleanData)):, -1]

temp = []
for i in Y_test:
    temp.append([1, 0]) if i == 2 else temp.append([0, 1])

Y_test = np.array(temp)

X = tf.placeholder(tf.float32, [None, 4], name="Input")
lr = tf.placeholder(tf.float32, name="learning_rate")
tf.summary.scalar("Learning_rate", lr)

W1 = tf.Variable(tf.truncated_normal([4, 50], stddev=0.1), name="W1")
b1 = tf.Variable(tf.zeros([50]), name="B1")

W2 = tf.Variable(tf.truncated_normal([50, 2], stddev=0.1), name="W2")
b2 = tf.Variable(tf.zeros([2]), name="B2")

with tf.name_scope("First_Layer"):
    l1 = tf.nn.leaky_relu(tf.matmul(X, W1) + b1)

w1_hist = tf.summary.histogram("weights1", W1)
b1_hist = tf.summary.histogram("biases1", b1)
l1_hist = tf.summary.histogram("activations_1", l1)

with tf.name_scope("Second_Layer"):
    l2 = tf.matmul(l1, W2) + b2

w2_hist = tf.summary.histogram("weights2", W2)
b2_hist = tf.summary.histogram("biases2", b2)

with tf.name_scope("Output"):
    Y = tf.nn.softmax(l2)

Y_ = tf.placeholder(tf.float32, [None, 2], name="labels")

# cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=l2,
#                                                         labels=Y_)
# cross_entropy = tf.reduce_mean(cross_entropy)*613

with tf.name_scope("Train_step"):
    cross_entropy = -tf.reduce_sum(Y_*tf.log(Y))
    is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

cost_train = tf.summary.scalar("cost_train", cross_entropy)
acc_train = tf.summary.scalar("acc_train", accuracy)

with tf.name_scope("Test_step"):
    cross_entropy_test = -tf.reduce_sum(Y_*tf.log(Y))
    is_correct_test = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy_test = tf.reduce_mean(tf.cast(is_correct_test, tf.float32))

cost_test = tf.summary.scalar("cost_test", cross_entropy_test)
acc_test = tf.summary.scalar("acc_test", accuracy_test)

optimizer = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

merged_train_summary_op = tf.summary.merge([w1_hist, w2_hist, l1_hist, b1_hist, b2_hist, cost_train, acc_train])
merged_test_summary_op = tf.summary.merge([acc_test, cost_test])


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    writer_train = tf.summary.FileWriter('./log/train')
    writer_test = tf.summary.FileWriter('./log/test')
    writer_train.add_graph(sess.graph)
    writer_test.add_graph(sess.graph)

    # 500 epochs
    for i in range(1000):
        # learning rate decay
        max_learning_rate = 0.003
        min_learning_rate = 0.00001
        decay_speed = 2000.0  # 0.003-0.0001-2000=>0.9826 done in 5000 iterations
        learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)

        acc, ce = sess.run([accuracy, cross_entropy], feed_dict={X: X_train,
                                                                 Y_: Y_train})
        if i%10 == 0:
            print("Accuracy train:", acc, "Train loss", ce)

        acc, ce = sess.run([accuracy_test, cross_entropy_test], feed_dict={X: X_test,
                                                                 Y_: Y_test})
        if i%10 == 0:
            print("Accuracy test:", acc, "Test loss", ce)

        _, summary_train, summary_test = sess.run([optimizer, merged_train_summary_op, merged_test_summary_op], feed_dict={X: X_train,
                                                                         Y_: Y_train,
                                                                         lr: learning_rate})
        writer_train.add_summary(summary_train, i)
        writer_test.add_summary(summary_test, i)
        # writer_test.add_summary(cost_test, i)
