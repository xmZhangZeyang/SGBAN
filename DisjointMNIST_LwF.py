import tensorflow as tf
import numpy as np


mnist0to4 = np.load('DisjointMNIST/mnist0to4.npz')
mnist5to9 = np.load('DisjointMNIST/mnist5to9.npz')

train_set_59 = mnist5to9["train_data"].tolist()
train_set_labels_59 = mnist5to9["train_labels"].tolist()
test_set_59 = mnist5to9["test_data"].tolist()
test_set_labels_59 = mnist5to9["test_labels"].tolist()

train_set_04 = mnist0to4["train_data"].tolist()
train_set_labels_04 = mnist0to4["train_labels"].tolist()
test_set_04 = mnist0to4["test_data"].tolist()
test_set_labels_04 = mnist0to4["test_labels"].tolist()


model_path = "Models/"
batch_size = 1000


def init_network(architecture):
    ph_input = tf.placeholder(dtype=tf.float32, shape=[None, architecture[0]])
    ph_label04 = tf.placeholder(dtype=tf.float32, shape=[None, architecture[-1]])
    ph_label59 = tf.placeholder(dtype=tf.float32, shape=[None, architecture[-1]])
    fc_list = [ph_input]
    fc_stop_list = [ph_input]
    k_list = []
    b_list = []
    para_sum = tf.Variable(0, dtype=tf.float32)
    for i in range(len(architecture) - 2):
        k = tf.Variable(tf.random.normal([architecture[i], architecture[i+1]]), dtype=tf.float32, trainable=True)
        b = tf.Variable(tf.random.normal([architecture[i+1]]), dtype=tf.float32, trainable=True)
        k_list.append(k)
        b_list.append(b)
        para_sum = tf.add(para_sum, tf.reduce_sum(tf.square(k)))
        para_sum = tf.add(para_sum, tf.reduce_sum(tf.square(b)))
        fc_list.append(tf.nn.sigmoid(tf.add(tf.matmul(fc_list[-1], k), b)))
        # stop gradients for the Warm_Up
        k_stop = tf.stop_gradient(k)
        b_stop = tf.stop_gradient(b)
        fc_stop_list.append(tf.nn.sigmoid(tf.add(tf.matmul(fc_stop_list[-1], k_stop), b_stop)))
    k1 = tf.Variable(tf.random.normal([architecture[-2], architecture[-1]]), dtype=tf.float32, trainable=True)
    b1 = tf.Variable(tf.random.normal([architecture[-1]]), dtype=tf.float32, trainable=True)
    k_list.append(k1)
    b_list.append(b1)
    para_sum = tf.add(para_sum, tf.reduce_sum(tf.square(k1)))
    para_sum = tf.add(para_sum, tf.reduce_sum(tf.square(b1)))
    logits1 = tf.add(tf.matmul(fc_list[-1], k1), b1)
    k2 = tf.Variable(tf.random.normal([architecture[-2], architecture[-1]]), dtype=tf.float32, trainable=True)
    b2 = tf.Variable(tf.random.normal([architecture[-1]]), dtype=tf.float32, trainable=True)
    k_list.append(k2)
    b_list.append(b2)
    para_sum = tf.add(para_sum, tf.reduce_sum(tf.square(k2)))
    para_sum = tf.add(para_sum, tf.reduce_sum(tf.square(b2)))
    logits2 = tf.add(tf.matmul(fc_list[-1], k2), b2)
    softmax1 = tf.nn.softmax(logits1)
    softmax2 = tf.nn.softmax(logits2)
    logits_warmup = tf.nn.sigmoid(tf.add(tf.matmul(fc_stop_list[-1], k2), b2))
    return ph_input, ph_label04, ph_label59, para_sum, logits1, logits_warmup, logits2, softmax1, softmax2, k_list, b_list


# define network
sess = tf.InteractiveSession()
architecture = [784, 79, 11, 5]
ph_input, ph_label04, ph_label59, para_sum, logits1, logits_warmup, logits2, softmax1, softmax2, k_list, b_list \
    = init_network(architecture)

# eval task 1
index1 = tf.reshape(tf.constant(np.arange(5139, dtype=np.int64)), shape=[5139, 1])
labels1 = tf.reshape(tf.argmax(softmax1, axis=1), shape=[5139, 1])
concat1 = tf.concat([index1, labels1], axis=1)
onehot1 = tf.sparse_to_dense(concat1, [5139, 5], 1.0, 0.0)
error1 = tf.reduce_sum(tf.square(onehot1 - ph_label04)) / 2
# eval task 2
index2 = tf.reshape(tf.constant(np.arange(4861, dtype=np.int64)), shape=[4861, 1])
labels2 = tf.reshape(tf.argmax(softmax2, axis=1), shape=[4861, 1])
concat2 = tf.concat([index2, labels2], axis=1)
onehot2 = tf.sparse_to_dense(concat2, [4861, 5], 1.0, 0.0)
error2 = tf.reduce_sum(tf.square(onehot2 - ph_label59)) / 2

rr = 5e-7 * para_sum
ld = 1.0
loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits1, labels=ph_label04)) + rr
loss_warmup = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_warmup, labels=ph_label59)) \
              + 5e-7 * (tf.reduce_sum(tf.square(k_list[-1])) + tf.reduce_sum(tf.square(b_list[-1])))
loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits2, labels=ph_label59)) + rr

yo_hat = tf.sqrt(softmax1) / tf.expand_dims(tf.reduce_sum(tf.sqrt(softmax1), axis=1), axis=1)
yo = tf.sqrt(ph_label04) / tf.expand_dims(tf.reduce_sum(tf.sqrt(ph_label04), axis=1), axis=1)
loss_old = -ld * tf.reduce_mean(tf.multiply(yo, tf.log(yo_hat)))
loss_lwf = loss_old + loss2

train_task1 = tf.train.AdamOptimizer(1e-4).minimize(loss1)
warm_up = tf.train.AdamOptimizer(1e-4).minimize(loss_warmup)
train_task2 = tf.train.AdamOptimizer(1e-5).minimize(loss_lwf)
saver = tf.train.Saver()
tf.global_variables_initializer().run()
# saver.restore(sess, "Models/model_lwf")


# train task 1
for i in range(25000):
    random_data = []
    random_labels = []
    for ri in range(batch_size):
        r = np.random.randint(0, len(train_set_labels_04))
        random_data.append(train_set_04[r])
        random_labels.append(train_set_labels_04[r])
    sess.run(train_task1, feed_dict={ph_input: random_data, ph_label04: random_labels})
    if i % 100 == 99:
        lo1, er1 = sess.run([loss1, error1], feed_dict={ph_input: test_set_04, ph_label04: test_set_labels_04})
        print(i, "loss:", lo1, "accuracy:", (5139 - er1) / 5139)


# warm_up
for i in range(25000):
    random_data = []
    random_labels = []
    for ri in range(batch_size):
        r = np.random.randint(0, len(train_set_labels_59))
        random_data.append(train_set_59[r])
        random_labels.append(train_set_labels_59[r])
    sess.run(warm_up, feed_dict={ph_input: random_data, ph_label59: random_labels})
    if i % 100 == 99:
        lo2, er2 = sess.run([loss2, error2], feed_dict={ph_input: test_set_59, ph_label59: test_set_labels_59})
        print(i, "TASK_2 loss:", lo2, "accuracy:", (4861 - er2) / 4861)


# train task 2
yo_labels = sess.run(softmax1, feed_dict={ph_input: train_set_59})
# # save fake labels
# np.save("DisjointMNIST/y_o.npy", yo_labels)
# yo_labels = np.load("DisjointMNIST/y_o.npy").tolist()
for i in range(50000):
    random_data = []
    random_labels = []
    random_labels_o = []
    for ri in range(batch_size):
        r = np.random.randint(0, len(train_set_labels_59))
        random_data.append(train_set_59[r])
        random_labels.append(train_set_labels_59[r])
        random_labels_o.append(yo_labels[r])
    sess.run(train_task2, feed_dict={ph_input: random_data, ph_label59: random_labels, ph_label04: random_labels_o})
    if i % 100 == 99:
        lo1, er1 = sess.run([loss_old, error1], feed_dict={ph_input: test_set_04, ph_label04: test_set_labels_04})
        print(i, "TASK_1 loss:", lo1, "accuracy:", (5139 - er1) / 5139)
        lo2, er2 = sess.run([loss2, error2], feed_dict={ph_input: test_set_59, ph_label59: test_set_labels_59})
        print(i, "TASK_2 loss:", lo2, "accuracy:", (4861 - er2) / 4861)

save_path = saver.save(sess, "Models/model_lwf")
print("Saved!")
tf.reset_default_graph()
sess.close()
