import tensorflow as tf
import numpy as np


def sign_zto(tensor):
    return (tf.sign(tensor) + 1.0) * 0.5


class Fully_Connected_Network:
    def __init__(self, architecture, activation=None, dtype=tf.float32):
        input_dims = architecture[0]
        output_dims = architecture[-1]
        paras_count = 0
        for i in range(len(architecture) - 1):
            paras_count += (architecture[i] + 1) * architecture[i + 1]
        self.sess = tf.InteractiveSession()
        self.layer_in = tf.placeholder(dtype=dtype, shape=[None, input_dims])
        self.labels = tf.placeholder(dtype=dtype, shape=[None, output_dims])
        self.lr = tf.placeholder(dtype=dtype)
        self.pos_weight = tf.placeholder(dtype=dtype)
        # self.pc = tf.placeholder(dtype=dtype)
        # self.nc = tf.placeholder(dtype=dtype)
        self.ld = tf.placeholder(dtype=dtype)
        self.networks = [self.layer_in]
        self.networks_sign = [self.layer_in]
        self.w_list = []
        self.b_list = []
        self.norm_opt = []
        self.double_opt = []
        w_length = []
        self.logits = None
        self.loss_p = None
        self.loss_n = None
        if activation is None:
            activation = list()
            for i in range(len(architecture) - 1):
                activation.append(None)
        else:
            for ai in range(len(activation)):
                if activation[ai] == "sigmoid":
                    activation[ai] = tf.nn.sigmoid
                elif activation[ai] == "relu":
                    activation[ai] = tf.nn.relu
                elif activation[ai] == "softmax":
                    activation[ai] = tf.nn.softmax
                elif activation[ai] == "tanh":
                    activation[ai] = tf.nn.tanh
                elif activation[ai] is None:
                    pass
                else:
                    print("Activation is invalid")
                    activation[ai] = None
        for i in range(len(architecture) - 1):
            self.w_list.append(tf.Variable(tf.random.normal([architecture[i], architecture[i + 1]],
                                                            mean=0.0, stddev=0.01), dtype=tf.float32, trainable=True))
            self.b_list.append(tf.Variable(tf.random.normal([architecture[i + 1]],
                                                            mean=0.0, stddev=0.01), dtype=tf.float32, trainable=True))
            # self.b_norm.append(b_list[-1].assign(
            #     tf.divide(b_list[-1], tf.sqrt(tf.reduce_sum(tf.square(w_list[-1]), axis=0)))))
            # self.w_norm.append(w_list[-1].assign(
            #     tf.divide(w_list[-1], tf.sqrt(tf.reduce_sum(tf.square(w_list[-1]), axis=0)))))
            if i == len(architecture) - 2:
                self.logits = tf.matmul(self.networks[-1], self.w_list[-1]) + self.b_list[-1]
                self.sigmoid = tf.nn.sigmoid(self.logits)
                self.pos_sign = sign_zto(self.logits)

                self.logits_sign = tf.matmul(self.networks_sign[-1], self.w_list[-1]) + self.b_list[-1]
            self.networks.append(activation[i](tf.matmul(self.networks[-1], self.w_list[-1]) + self.b_list[-1]))
            self.networks_sign.append(sign_zto(tf.matmul(self.networks_sign[-1], self.w_list[-1]) + self.b_list[-1]))
        for i in range(len(self.b_list)):
            w_length.append(tf.sqrt(tf.reduce_sum(tf.square(self.w_list[i]), axis=0)))
            self.norm_opt.append(self.b_list[i].assign(tf.divide(self.b_list[i], w_length[i])))
            self.norm_opt.append(self.w_list[i].assign(tf.divide(self.w_list[i], w_length[i])))
        for i in range(len(self.b_list)):
            self.double_opt.append(self.b_list[i].assign(tf.multiply(self.b_list[i], 2.0)))
            self.double_opt.append(self.w_list[i].assign(tf.multiply(self.w_list[i], 2.0)))

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=self.logits))
        # ############################################################################################################
        self.var_list = tf.trainable_variables()
        self.grad_list = tf.gradients(self.loss, self.var_list)
        self.grad_var = []
        for i in range(len(self.var_list)):
            self.grad_var.append((self.grad_list[i], self.var_list[i]))
        self.train_opt = tf.train.AdamOptimizer(learning_rate=self.lr).apply_gradients(self.grad_var)
        # #################################################################################################
        self.grad_1_dim_ph = tf.placeholder(dtype=dtype, shape=[paras_count])
        self.var_list = tf.trainable_variables()
        point = 0
        self.grad_update = []
        for i in range(len(self.var_list)):
            var_shape = self.var_list[i].get_shape()
            dim = 1
            for di in var_shape:
                dim *= di
            grad = tf.reshape(self.grad_1_dim_ph[point:point + dim], var_shape)
            point += dim
            self.grad_update.append((grad, self.var_list[i]))
        self.train_with_grads = tf.train.AdamOptimizer(learning_rate=self.lr).apply_gradients(self.grad_update)

        # self.train_opt_momentum = tf.train.MomentumOptimizer(1e-4, 0.618, use_nesterov=True).minimize(self.loss)
        self.saver = tf.train.Saver()
        tf.global_variables_initializer().run()

    def training(self, data, labels, lr):
        _, lo = self.sess.run([self.train_opt, self.loss],
                              feed_dict={self.layer_in: data, self.labels: labels, self.lr: lr})
        return lo

    def get_grads(self, data, labels):
        grads_value = self.sess.run(self.grad_list, feed_dict={self.layer_in: data, self.labels: labels})
        return grads_value

    def apply_grads(self, grads, lr):
        self.sess.run(self.train_with_grads, feed_dict={self.grad_1_dim_ph: grads, self.lr: lr})

    def get_last_w(self):
        w = self.sess.run(self.w_list[-1])
        return np.sum(np.square(w))

    def get_loss(self, data, labels):
        lo = self.sess.run(self.loss, feed_dict={self.layer_in: data, self.labels: labels})
        return lo

    def double_all_weight(self):
        self.sess.run(self.double_opt)

    def weight_norm(self):
        self.sess.run(self.norm_opt)

    def sigmoid(self, pred):
        pred_sigmoid = tf.nn.sigmoid(pred)
        output = self.sess.run(pred_sigmoid)
        return output

    def softmax(self, pred):
        pred_sigmoid = tf.nn.softmax(pred)
        output = self.sess.run(pred_sigmoid)
        return output

    def prediction(self, data, labels):
        pred = self.sess.run(self.logits, feed_dict={self.layer_in: data})
        pred_onehot = np.zeros([pred.shape[0], pred.shape[1]])
        pred_zero_sign = np.maximum(np.sign(pred), 0)
        for pi in range(pred.shape[0]):
            am = np.argmax(pred[pi])
            pred_onehot[pi, am] = 1
        counting_correct_soft = int(pred.shape[0] - np.sum(np.absolute(labels - pred_onehot)) / 2)
        counting_correct_sign = 0
        sign_diff = np.sum(np.square(pred_zero_sign - labels), axis=1)
        for pi in range(pred.shape[0]):
            if sign_diff[pi] == 0:
                counting_correct_sign += 1
        return pred, counting_correct_soft, counting_correct_sign

    def prediction_binary(self, data, labels):
        pred = self.sess.run(self.logits_sign, feed_dict={self.layer_in: data})
        pred_onehot = np.zeros([pred.shape[0], pred.shape[1]])
        pred_zero_sign = np.maximum(np.sign(pred), 0)
        for pi in range(pred.shape[0]):
            am = np.argmax(pred[pi])
            pred_onehot[pi, am] = 1
        counting_correct_soft = int(pred.shape[0] - np.sum(np.absolute(labels - pred_onehot)) / 2)
        counting_correct_sign = 0
        sign_diff = np.sum(np.square(pred_zero_sign - labels), axis=1)
        for pi in range(pred.shape[0]):
            if sign_diff[pi] == 0:
                counting_correct_sign += 1
        return pred, counting_correct_soft, counting_correct_sign

    def correct_pw(self, data, labels):
        logits = self.sess.run(self.logits, feed_dict={self.layer_in: data})
        pred = np.maximum(0, np.sign(logits))
        p = 0
        n = 0
        for i in range(len(labels)):
            if labels[i][0] == 1:
                if pred[i][0] == 1:
                    p += 1
            else:
                if pred[i][0] == 0:
                    n += 1
        return p, n

    def loss_weight(self, data, labels):
        loss_1, loss_2 = self.sess.run([self.loss_p, self.loss_n], feed_dict={self.layer_in: data, self.labels: labels})
        return loss_1, loss_2

    def save(self, path):
        self.saver.save(self.sess, path)
        # print("Model Saved!")

    def restore(self, path):
        self.saver.restore(self.sess, path)
        print("Model Restored!")
#
# 简要说明：
# 使用分3步：1、初始化网络结构（调用构造函数）。  2、设置训练损失和优化器。   3、训练（feed数据和标签给网络优化器）
# 这三步对应了3个函数，可以修改training函数的输出来调试更多信息，需要注意每个要输出的值都需要先运行sess.run
# 比如这里sess.run了[self.train_opt, self.training_loss, self.networks]3个东西，函数就按顺序返回这3个东西，
# self.train_opt是训练操作，这个不能删，返回值其实是训练状态，没啥用，所以用“_”接收丢弃
# 需要返回什么数据就追加到self.sess.run([])里面就行
#
