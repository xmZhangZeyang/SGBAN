import tensorflow as tf
import numpy as np


class Network:
    def __init__(self, architecture, lamda, model=None):
        # init parameters
        self.architecture = architecture
        self.sess = tf.InteractiveSession()
        self.ph_input = tf.placeholder(dtype=tf.float32, shape=[None, self.architecture[0]])
        self.ph_label = tf.placeholder(dtype=tf.float32, shape=[None, self.architecture[-1]])
        self.ph_lr = tf.placeholder(dtype=tf.float32, shape=[1])
        self.k_list = []
        self.b_list = []
        if model is None:
            for ai in range(len(architecture) - 1):
                self.k_list.append(tf.Variable(
                    tf.random.normal([architecture[ai], architecture[ai + 1]]), dtype=tf.float32, trainable=True))
                self.b_list.append(tf.Variable(tf.random.normal([architecture[ai + 1]]), dtype=tf.float32, trainable=True))
            self.var_star = []
            self.fisher_list = []
            self.fisher_num = 0
        else:
            # read model
            self.model = np.load(model)
            _k_list, _b_list = self.model["k_list"], self.model["b_list"]
            for ai in range(len(self.architecture) - 1):
                self.k_list.append(tf.Variable(_k_list[ai], dtype=tf.float32, trainable=True))
                self.b_list.append(tf.Variable(_b_list[ai], dtype=tf.float32, trainable=True))
            self.var_star = self.model["var_star"]
            self.fisher_list = self.model["fisher"]
            self.fisher_num = self.model["num"]
        self.var_list = self.k_list + self.b_list
        if len(self.fisher_list) == 0:
            for vi in range(len(self.var_list)):
                self.fisher_list.append(np.zeros(self.var_list[vi].get_shape().as_list()))
        if len(self.var_star) == 0:
            for vi in range(len(self.var_list)):
                self.var_star.append(np.zeros(self.var_list[vi].get_shape().as_list()))
        # init network
        self.net = [self.ph_input]
        self.logits = None
        for ai in range(len(self.architecture) - 1):
            if ai != len(self.architecture) - 2:
                self.net.append(tf.nn.sigmoid(tf.add(tf.matmul(self.net[-1], self.k_list[ai]), self.b_list[ai])))
            else:
                self.logits = tf.add(tf.matmul(self.net[-1], self.k_list[ai]), self.b_list[ai])
                self.net.append(tf.nn.softmax(self.logits))
        self.probs = self.net[-1]
        self.class_ind = tf.to_int32(tf.multinomial(tf.log(self.probs), 1)[0][0])
        self.grad = tf.gradients(tf.log(self.probs[0, self.class_ind]), self.var_list)
        # self.grad2 = tf.gradients(self.grad, self.var_list)
        # self.grad2 = []
        # for vi in range(len(self.var_list)):
        #     self.grad2.append(tf.gradients(self.grad[vi], self.var_list[vi])[0])

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.ph_label))
        self.ewc_loss = self.loss
        if self.fisher_num != 0:
            for v in range(len(self.var_list)):
                self.ewc_loss += (lamda / (2 * self.fisher_num)) * tf.reduce_sum(
                    tf.multiply(self.fisher_list[v].astype(np.float32), tf.square(self.var_list[v] - self.var_star[v])))
        self.train_opt = tf.train.AdamOptimizer(self.ph_lr[0]).minimize(self.ewc_loss)
        tf.global_variables_initializer().run()

    def clear_fisher(self):
        self.fisher_list = []
        for vi in range(len(self.var_list)):
            self.fisher_list.append(np.zeros(self.var_list[vi].get_shape().as_list()))
        self.fisher_num = 0

    def update_fisher(self, data, loops):
        # probs = self.net[-1]
        # class_ind = tf.to_int32(tf.multinomial(tf.log(probs), 1)[0][0])
        for di in range(loops):
            # r_int = np.random.randint(len(data))
            for r_int in range(len(data)):
                ders = self.sess.run(self.grad, feed_dict={self.ph_input: data[r_int:r_int + 1]})
                for vi in range(len(self.fisher_list)):
                    self.fisher_list[vi] += np.square(ders[vi])
            print(np.sum(self.fisher_list[0]))
        self.fisher_num += loops * len(data)
        # for vi in range(len(self.fisher_list)):
        #     self.fisher_list[vi] /= num_samples

    # def test_fisher(self):
    #     print(np.sum(self.fisher_list[0]))
    #     for fi in range(len(self.fisher_list)):
    #         self.fisher_list[fi] = np.minimum(np.maximum((-1) * self.fisher_list[fi], 0), 50)
    #     print(np.sum(self.fisher_list[0]))

    def set_star(self):
        _k_list, _b_list = self.sess.run([self.k_list, self.b_list])
        self.var_star = _k_list + _b_list

    def eval(self, test_set, test_labels):
        num = len(test_set)
        index = tf.reshape(tf.constant(np.arange(num, dtype=np.int64)), shape=[num, 1])
        labels = tf.reshape(tf.argmax(self.net[-1], axis=1), shape=[num, 1])
        concat = tf.concat([index, labels], axis=1)
        one_hot = tf.sparse_to_dense(concat, [num, self.architecture[-1]], 1.0, 0.0)
        error = tf.reduce_sum(tf.square(one_hot - self.ph_label)) / 2
        accuracy = (num - error) / num
        err, acc = self.sess.run([error, accuracy], feed_dict={self.ph_input: test_set, self.ph_label: test_labels})
        print("error:", err, ", accuracy:", acc)
        return err, acc

    def train(self, train_set, train_labels, loop, batchsize, lr):
        for li in range(loop):
            random_data = []
            random_labels = []
            for ri in range(batchsize):
                r = np.random.randint(0, len(train_set))
                random_data.append(train_set[r])
                random_labels.append(train_labels[r])
            self.sess.run(self.train_opt,
                          feed_dict={self.ph_input: random_data, self.ph_label: random_labels, self.ph_lr: [lr]})

    def save(self, path):
        _k_list, _b_list = self.sess.run([self.k_list, self.b_list])
        np.savez(path, k_list=_k_list, b_list=_b_list, var_star=self.var_star,
                 fisher=self.fisher_list, num=self.fisher_num)

    def close(self):
        tf.reset_default_graph()
        self.sess.close()
