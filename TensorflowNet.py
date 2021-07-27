import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
# import copy
import os


os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'
# np.seterr(divide='ignore', invalid='ignore')


ndl = []


def data_norm(mnist_image):
    output = []
    for di in range(len(mnist_image)):
        temp = np.delete(np.array(mnist_image[di]), ndl)
        temp[temp > 0.3] = 1
        temp[temp < 0.3] = -1
        output.append(temp.tolist())
    return output


# def relu_sphere(tensor):
#     # return tf.multiply(tensor, tf.expand_dims(tf.divide(1, tf.sqrt(tf.reduce_sum(tf.square(tensor), axis=1))), -1))
#     norm = tf.expand_dims(tf.divide(1, tf.sqrt(tf.reduce_sum(tf.square(tf.nn.relu(tensor)), axis=1))), -1)
#     for ni in norm:
#         if tf.is_inf(ni[0]):
#             tf.assign(ni[0], 0)
#     return tf.multiply(tf.nn.relu(tensor), norm)


def relu_sphere(tensor, batchsize):
    return tf.multiply(tensor, tf.expand_dims(tf.divide(1, tf.sqrt(tf.reduce_sum(tf.square(tensor), axis=1))), -1))
    # one_list = tf.ones([batchsize, 1], dtype=tf.float32)
    # nz_tensor = tf.concat([one_list, tf.nn.relu(tensor)], axis=1)
    # norm = tf.expand_dims(tf.divide(1, tf.sqrt(tf.reduce_sum(tf.square(nz_tensor), axis=1))), -1)
    # return tf.multiply(nz_tensor, norm)


def zero_sign(tensor):
    return tf.maximum(tf.sign(tensor), 0)


def zero_softsign(tensor):
    return (tf.nn.tanh(tensor) + 1) / 2


def Rel_GW(labels, logits):
    # pos_l2 = (tf.nn.l2_normalize(logits, axis=1) + 1) * 0.5
    pos_sign = (tf.sign(logits) + 1.0) * 0.5
    loss_p = tf.reduce_sum(tf.multiply(pos_sign, tf.multiply(
             labels, tf.div(1.0, (1 + tf.exp(logits))))))
    loss_n = tf.reduce_sum(tf.multiply(1 - pos_sign, tf.multiply(
             1 - labels, tf.div(1.0, (1 + tf.exp(-logits))))))
    pc = tf.stop_gradient(loss_p)
    nc = tf.stop_gradient(loss_n)

    loss_inc = tf.reduce_mean(tf.multiply(logits, (pos_sign - labels)))
    loss_cor = - tf.reduce_mean(
               2 * nc / tf.clip_by_value((pc + nc), 1e-15, np.inf) *
               tf.multiply(pos_sign, tf.multiply(
                   labels, tf.log(tf.clip_by_value(tf.nn.sigmoid(logits), 1e-15, np.inf)))) +
               2 * pc / tf.clip_by_value((pc + nc), 1e-15, np.inf) *
               tf.multiply(1 - pos_sign, tf.multiply(
                   1 - labels, tf.log(tf.clip_by_value(1 - tf.nn.sigmoid(logits), 1e-15, np.inf)))))

    loss = loss_inc + 1.0 * loss_cor
    return loss


def focal_loss(labels, logits):
    sigmoid = tf.clip_by_value(tf.nn.sigmoid(logits), clip_value_min=1e-7, clip_value_max=1-1e-7)
    loss = -tf.reduce_mean(tf.multiply(tf.pow(1 - sigmoid, 16),
                                       tf.multiply(labels, tf.log(sigmoid)))
                           + tf.multiply(tf.pow(sigmoid, 16),
                                         tf.multiply(1 - labels, tf.log(1 - sigmoid))))
    # loss = -tf.reduce_mean(tf.multiply(labels, tf.log(sigmoid))
    #                        + tf.multiply(1 - labels, tf.log(1 - sigmoid)))
    return loss


# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# train_set = data_norm(mnist.train.images)
# train_set_labels = mnist.train.labels
# train_set_labels[train_set_labels > 0] = 1
# train_set_labels[train_set_labels <= 0] = 0
# train_set_labels = train_set_labels.tolist()
# test_set = data_norm(mnist.test.images)
# test_set_labels = mnist.test.labels
# test_set_labels[test_set_labels > 0] = 1
# test_set_labels[test_set_labels <= 0] = 0
# test_set_labels = test_set_labels.tolist()


# def train(architecture, ws, bs, data, labels, loop):
#     _data = np.array(copy.deepcopy(data))
#     _labels = np.array(copy.deepcopy(labels))
#     _labels[_labels <= 0] = 0
#     _labels[_labels > 0] = 1
#     sess = tf.InteractiveSession()
#     ph_input = tf.placeholder(dtype=tf.float32, shape=[None, architecture[0]])
#     ph_label = tf.placeholder(dtype=tf.float32, shape=[None, architecture[-1]])
#     net = [ph_input]
#     w_list = []
#     b_list = []
#     for i in range(len(architecture) - 1):
#         w_list.append(tf.Variable(ws[i], dtype=tf.float32, trainable=True))
#         b_list.append(tf.Variable(bs[i], dtype=tf.float32, trainable=True))
#         net.append(tf.matmul(relu_sphere(net[-1], _data.shape[0]), w_list[-1]) + b_list[-1])
#     net_output = relu_sphere(net[-1], _data.shape[0])[:, 1:]
#
#     loss = tf.nn.softmax_cross_entropy_with_logits(logits=net[-1], labels=ph_label)
#     ##############################################################################################
#     # t_add_f = np.ones(_labels.shape[1]) * _labels.shape[0]
#     # t_only = np.maximum(np.sum(_labels, axis=0), 0.1)
#     # f_div_t = np.divide(t_add_f - t_only, t_only)
#     # # print(t_add_f, t_only, f_div_t)
#     # # f_div_t = np.ones(_labels.shape[1])
#     # loss_list = []
#     # for i in range(len(f_div_t)):
#     #     loss_list.append(tf.nn.weighted_cross_entropy_with_logits(ph_label[:, i], net_output[:, i], f_div_t[i]))
#     # loss = tf.reduce_sum(loss_list)
#     ##############################################################################################
#     train_opt = tf.train.AdamOptimizer(1e-3).minimize(loss)
#     tf.global_variables_initializer().run()
#     ##############################################################################################
#     for eps in range(loop):
#         sess.run(train_opt, feed_dict={ph_input: _data, ph_label: _labels})
#         ############################################################################################
#         if eps % 10 == 9:
#             pred = sess.run(net_output, feed_dict={ph_input: _data})
#             pred = np.array(pred)
#             hot = np.argmax(pred, axis=1)
#             one_hot = np.zeros([pred.shape[0], pred.shape[1]])
#             for hi in range(len(hot)):
#                 one_hot[hi, hot[hi]] = 1
#             counting_correct = 0
#             for pi in range(pred.shape[0]):
#                 if np.sum(np.absolute(_labels[pi] - one_hot[pi])) == 0:
#                     counting_correct += 1
#             print('accuracy:', counting_correct, pred.shape[0])
#     ###########################################################################################
#     w_new, b_new = sess.run([w_list, b_list])
#     # normalization
#     # w_norm = []
#     # b_norm = []
#     # for i in range(len(b_new)):
#     #     norm = 1 / np.sqrt(np.sum(np.square(w_new[i]), axis=0))
#     #     w_norm.append(np.multiply(w_new[i], norm))
#     #     b_norm.append(np.multiply(b_new[i], norm))
#     tf.reset_default_graph()
#     sess.close()
#     return w_new, b_new    # w_norm, b_norm


def fine_tune(architecture, ws, bs, data, labels, batch_size=1000, loop=100, lr=1e-6, mask=None):
    # if batch_size > len(data):
    #     batch_size = len(data)
    if mask is not None:
        print("mask:", mask)
    _data = np.array(data)
    _labels = np.array(labels)
    # _data = np.array(copy.deepcopy(data))
    # _labels = np.array(copy.deepcopy(labels))
    # _labels[_labels <= 0] = 0
    # _labels[_labels > 0] = 1
    sess = tf.InteractiveSession()
    ph_input = tf.placeholder(dtype=tf.float32, shape=[None, architecture[0]])
    if mask is not None:
        ph_label = tf.placeholder(dtype=tf.float32, shape=[None, architecture[-1] - mask[-1]])
        ph_weighted = tf.placeholder(dtype=tf.float32, shape=architecture[-1] - mask[-1])
    else:
        ph_label = tf.placeholder(dtype=tf.float32, shape=[None, architecture[-1]])
        ph_weighted = tf.placeholder(dtype=tf.float32, shape=architecture[-1])
    net_sign = [ph_input]
    net_sigmoid = [ph_input]
    w_list = []
    b_list = []
    logits = None
    for i in range(len(architecture) - 1):
        if mask is None:
            w_list.append(tf.Variable(ws[i], dtype=tf.float32, trainable=True))
            b_list.append(tf.Variable(bs[i], dtype=tf.float32, trainable=True))
        else:
            w_u = tf.Variable(np.array(ws[i])[:, :mask[i]], dtype=tf.float32, trainable=False)
            w_t = tf.Variable(np.array(ws[i])[:, mask[i]:], dtype=tf.float32, trainable=True)
            b_u = tf.Variable(bs[i][:mask[i]], dtype=tf.float32, trainable=False)
            b_t = tf.Variable(bs[i][mask[i]:], dtype=tf.float32, trainable=True)
            w_concat = tf.concat([w_u, w_t], axis=1)
            b_concat = tf.concat([b_u, b_t], axis=0)
            w_list.append(w_concat)
            b_list.append(b_concat)
        if i == len(architecture) - 2:
            if mask is not None:
                logits = tf.add(tf.matmul(net_sigmoid[-1], w_list[-1]), b_list[-1])[:, mask[-1]:]
            else:
                logits = tf.add(tf.matmul(net_sigmoid[-1], w_list[-1]), b_list[-1])
        net_sigmoid.append(zero_softsign(tf.matmul(net_sigmoid[-1], w_list[-1]) + b_list[-1]))
        net_sign.append(zero_sign(tf.matmul(net_sign[-1], w_list[-1]) + b_list[-1]))
    # loss = tf.nn.softmax_cross_entropy_with_logits(logits=net[-1], labels=ph_label)
    ##############################################################################################
    # t_add_f = np.ones(len(_labels[0])) * batch_size
    loss_list = []
    for i in range(len(_labels[0])):
        loss_list.append(tf.nn.weighted_cross_entropy_with_logits(ph_label[:, i], logits[:, i], ph_weighted[i]))
        # loss_list.append(tf.nn.sigmoid_cross_entropy_with_logits(labels=ph_label[:, i:i+1], logits=logits[:, i:i+1]))
        # loss_list.append(Rel_GW(ph_label[:, i:i+1], logits[:, i:i+1]))
        # loss_list.append(focal_loss(ph_label[:, i], logits[:, i]))
    loss = tf.reduce_mean(loss_list)
    # loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=ph_label, logits=net_output)
    ##############################################################################################
    train_opt = tf.train.AdamOptimizer(lr).minimize(loss)
    # train_opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(loss)
    tf.global_variables_initializer().run()
    ##############################################################################################
    eps = 0
    while eps < loop:
        random = np.random.permutation(len(data))
        random_data_all = _data[random]
        random_labels_all = _labels[random]
        if len(data) < batch_size:
            random_data = random_data_all
            random_labels = random_labels_all
            eps += 1
            # t_only = np.clip(np.sum(random_labels, axis=0), a_min=0.01, a_max=np.inf)
            # f_div_t = np.divide(t_add_f - t_only, t_only)
            f_div_t = np.ones(_labels.shape[1])
            # # print(t_add_f, t_only, f_div_t)
            sess.run(train_opt, feed_dict={ph_input: random_data, ph_label: random_labels, ph_weighted: f_div_t})
        else:
            for bi in range(len(data) // batch_size):
                random_data = random_data_all[bi * batch_size:(bi+1) * batch_size]
                random_labels = random_labels_all[bi * batch_size:(bi+1) * batch_size]
                eps += 1
                # t_only = np.clip(np.sum(random_labels, axis=0), a_min=0.01, a_max=np.inf)
                # f_div_t = np.divide(t_add_f - t_only, t_only)
                f_div_t = np.ones(_labels.shape[1])
                # # print(t_add_f, t_only, f_div_t)
                sess.run(train_opt, feed_dict={ph_input: random_data, ph_label: random_labels, ph_weighted: f_div_t})
        ############################################################################################
        # if eps % 100 == 99:
        #     pred = sess.run(net_output, feed_dict={ph_input: random_data})
        #     pred = np.array(pred)
        #     hot = np.argmax(pred, axis=1)
        #     one_hot = np.zeros([pred.shape[0], pred.shape[1]])
        #     for hi in range(len(hot)):
        #         one_hot[hi, hot[hi]] = 1
        #     counting_correct = 0
        #     for pi in range(pred.shape[0]):
        #         if np.sum(np.absolute(random_labels[pi] - one_hot[pi])) == 0:
        #             counting_correct += 1
        #     print('accuracy:', counting_correct, pred.shape[0])
    ###########################################################################################
    w_new, b_new = sess.run([w_list, b_list])
    tf.reset_default_graph()
    sess.close()
    return w_new, b_new    # w_norm, b_norm


def test(architecture, ws, bs, data, labels, softmax=False, softact=False, batch=1000, label_shift=None):
    _data = data
    _labels = labels
    sess = tf.InteractiveSession()
    ph_input = tf.placeholder(dtype=tf.float32, shape=[None, architecture[0]])
    net_sign = [ph_input]
    w_list = []
    b_list = []
    for i in range(len(architecture) - 1):
        w_list.append(tf.Variable(ws[i], dtype=tf.float32))
        b_list.append(tf.Variable(bs[i], dtype=tf.float32))
        if softact is False:
            if i == len(architecture) - 2:
                if softmax is True:
                    net_sign.append(tf.nn.softmax(tf.matmul(net_sign[-1], w_list[-1]) + b_list[-1]))
                else:
                    net_sign.append(zero_sign(tf.matmul(net_sign[-1], w_list[-1]) + b_list[-1]))
            else:
                net_sign.append(zero_sign(tf.matmul(net_sign[-1], w_list[-1]) + b_list[-1]))
        else:
            if i == len(architecture) - 2:
                if softmax is True:
                    net_sign.append(tf.nn.softmax(tf.matmul(net_sign[-1], w_list[-1]) + b_list[-1]))
                else:
                    net_sign.append(tf.nn.sigmoid(tf.matmul(net_sign[-1], w_list[-1]) + b_list[-1]))
            else:
                net_sign.append(tf.nn.sigmoid(tf.matmul(net_sign[-1], w_list[-1]) + b_list[-1]))
    if label_shift is not None:
        net_out = net_sign[-1][:, label_shift[0]:label_shift[1]]
    else:
        net_out = net_sign[-1]
    tf.global_variables_initializer().run()
    ##############################################################################################
    pred = []
    for i in range(int(np.ceil(len(_data) / batch))):
        if batch * (i + 1) <= len(_data):
            pred_i = sess.run(net_out, feed_dict={ph_input: _data[batch * i:batch * (i + 1)]})
        else:
            pred_i = sess.run(net_out, feed_dict={ph_input: _data[batch * i:]})
        if len(pred) == 0:
            pred = pred_i
        else:
            pred = np.concatenate((pred, pred_i), axis=0)
    # pred = sess.run(net_sign[-1], feed_dict={ph_input: _data})
    if softmax is True:
        pred = np.array(pred)
        hot = np.argmax(pred, axis=1)
        one_hot = np.zeros([pred.shape[0], pred.shape[1]])
        for hi in range(len(hot)):
            one_hot[hi, hot[hi]] = 1
    else:
        one_hot = np.array(pred)
        one_hot[one_hot > 0] = 1
        one_hot[one_hot <= 0] = 0
    append_list = []
    del_list = []
    for pi in range(pred.shape[0]):
        if np.sum(np.square(one_hot[pi] - _labels[pi])) == 0:
            append_list.append(pi)
        else:
            del_list.append(pi)
    print('test accuracy:', len(append_list), " / ", pred.shape[0])
    # net_list = sess.run(net, feed_dict={ph_input: _data})
    # print(net_list)
    tf.reset_default_graph()
    sess.close()
    return append_list, del_list


def forwarding(architecture, ws, bs, data, soft=False):
    _data = data
    sess = tf.InteractiveSession()
    ph_input = tf.placeholder(dtype=tf.float32, shape=[None, architecture[0]])
    net_sign = [ph_input]
    w_list = []
    b_list = []
    for i in range(len(architecture) - 1):
        w_list.append(tf.Variable(ws[i], dtype=tf.float32))
        b_list.append(tf.Variable(bs[i], dtype=tf.float32))
        if soft and i == len(architecture) - 2:
            net_sign.append(tf.nn.sigmoid(tf.matmul(net_sign[-1], w_list[-1]) + b_list[-1]))
        else:
            net_sign.append(zero_sign(tf.matmul(net_sign[-1], w_list[-1]) + b_list[-1]))
    tf.global_variables_initializer().run()
    ##############################################################################################
    pred = sess.run(net_sign, feed_dict={ph_input: _data})

    tf.reset_default_graph()
    sess.close()
    return pred


def find_target(architecture, ws, bs, data, label, ln, label_shift=None):
    _data = data
    _label = label
    sess = tf.InteractiveSession()
    ph_input = tf.placeholder(dtype=tf.float32, shape=[None, architecture[0]])
    net_sign = [ph_input]
    w_list = []
    b_list = []
    target = None
    for i in range(len(architecture) - 1):
        if i == ln:
            new_weight = tf.Variable(tf.zeros([1, architecture[ln + 1]]), dtype=tf.float32)
            old_weight = tf.Variable(ws[i], dtype=tf.float32)
            w_list.append(tf.concat([old_weight, new_weight], axis=0))
            b_list.append(tf.Variable(bs[i], dtype=tf.float32))
            target = tf.matmul(net_sign[-1], w_list[-1]) + b_list[-1]
        else:
            w_list.append(tf.Variable(ws[i], dtype=tf.float32))
            b_list.append(tf.Variable(bs[i], dtype=tf.float32))
        net_sign.append(zero_sign(tf.matmul(net_sign[-1], w_list[-1]) + b_list[-1]))
    tf.global_variables_initializer().run()
    ##############################################################################################
    pred, tar = sess.run([net_sign[-1], target], feed_dict={ph_input: _data})
    r_target = []
    print("the pred:",  pred, _label)
    for pi in range(len(pred)):
        if label_shift:
            hot_i = pred[pi][label_shift[0]:label_shift[1]]
        else:
            hot_i = pred[pi]
        if np.sum(np.square(hot_i - _label)) == 0:
            r_target.append(tar[pi])
    r_target = np.array(r_target)
    print(r_target.shape)
    target_v = np.sum(np.log(np.square(r_target) + 1), axis=1)
    pos = int(np.argmax(target_v))
    best_target = np.maximum(np.sign(r_target[pos]), 0)
    print("best_target", best_target, r_target[pos])

    tf.reset_default_graph()
    sess.close()
    return best_target


# def find_bottleneck(memory_patterns, patterns):
#     _data = np.concatenate((np.array(copy.deepcopy(memory_patterns)), np.array(copy.deepcopy(patterns))), axis=0)
#     _labels = np.concatenate((np.zeros([len(memory_patterns), 1]), np.ones([len(patterns), 1])), axis=0)
#
#     sess = tf.InteractiveSession()
#     ph_input = tf.placeholder(dtype=tf.float32, shape=[None, _data.shape[1]])
#     ph_label = tf.placeholder(dtype=tf.float32, shape=[None, 1])
#     w = tf.Variable(tf.random.normal([_data.shape[1] + 1, 1]), dtype=tf.float32, trainable=True)
#     b = tf.Variable(-0.1, dtype=tf.float32, trainable=True)
#     net = tf.nn.relu(tf.matmul(relu_sphere(ph_input, _data.shape[0]), w) + b)
#
#     loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=net, labels=ph_label))
#     # loss = - tf.reduce_sum(tf.multiply(net, ph_label))
#     train_opt = tf.train.AdamOptimizer(3e-3).minimize(loss)
#     tf.global_variables_initializer().run()
#     for i in range(12000):
#         sess.run(train_opt, feed_dict={ph_input: _data, ph_label: _labels})
#         if i % 100 == 0:
#             pred, k_out, b_out, lo = sess.run([net, w, b, loss], feed_dict={ph_input: _data, ph_label: _labels})
#             del_count = len(memory_patterns)
#             append_count = 0
#             # m = np.max(pred[0:len(memory_patterns)])
#             # pred -= m
#             for i in range(pred.shape[0]):
#                 if _labels[i, 0] > 0:
#                     if pred[i, 0] > 0:
#                         append_count += 1
#                 elif pred[i, 0] <= 0:
#                     del_count -= 1
#             print(append_count, del_count, lo)
#     pred, k_out, b_out = sess.run([net, w, b], feed_dict={ph_input: _data, ph_label: _labels})
#     del_list = []
#     append_list = []
#     # m = np.max(pred[0:len(memory_patterns)])
#     # pred -= m
#     for i in range(len(memory_patterns)):
#         if pred[i, 0] > 0:
#             del_list.append(i)
#     for i in range(len(memory_patterns), pred.shape[0]):
#         if pred[i, 0] > 0:
#             append_list.append(i - len(memory_patterns))
#     tf.reset_default_graph()
#     sess.close()
#     return append_list, del_list, k_out.tolist(), b_out.tolist()
#
#
# def bottleneck(architecture, ws, bs, data, labels, layer_n, memory_data, memory_labels):
#     _data = np.array(copy.deepcopy(data))
#     _labels = np.array(copy.deepcopy(labels))
#     _labels[_labels <= 0] = 0
#     _labels[_labels > 0] = 1
#     _memory_data = np.array(copy.deepcopy(memory_data))
#     _memory_labels = np.array(copy.deepcopy(memory_labels))
#     _memory_labels[_memory_labels <= 0] = 0
#     _memory_labels[_memory_labels > 0] = 1
#     sess = tf.InteractiveSession()
#     ph_input = tf.placeholder(dtype=tf.float32, shape=[None, architecture[0]])
#     ph_label = tf.placeholder(dtype=tf.float32, shape=[None, architecture[-1]])
#     net_data = [ph_input]
#     net_memory = [ph_input]
#     w_list = []
#     b_list = []
#     for i in range(len(architecture) - 1):
#         if i == layer_n:
#             w_list.append(tf.Variable(ws[i], dtype=tf.float32, trainable=True))
#             b_list.append(tf.Variable(bs[i], dtype=tf.float32, trainable=True))
#         else:
#             w_list.append(tf.Variable(ws[i], dtype=tf.float32, trainable=False))
#             b_list.append(tf.Variable(bs[i], dtype=tf.float32, trainable=False))
#         net_data.append(tf.matmul(relu_sphere(net_data[-1], _data.shape[0]), w_list[-1]) + b_list[-1])
#         net_memory.append(tf.matmul(relu_sphere(net_memory[-1], _memory_data.shape[0]), w_list[-1]) + b_list[-1])
#     net_data_output = relu_sphere(net_data[-1], _data.shape[0])[:, 1:]
#     net_memory_output = relu_sphere(net_memory[-1], _memory_data.shape[0])[:, 1:]
#
#     loss = tf.nn.softmax_cross_entropy_with_logits(logits=net_data_output, labels=ph_label)
#     train_opt = tf.train.AdamOptimizer(1e-3).minimize(loss)
#     tf.global_variables_initializer().run()
#     ##############################################################################################
#     pred = sess.run(net_data_output, feed_dict={ph_input: _data, ph_label: _labels})
#     pred = np.array(pred)
#     hot = np.argmax(pred, axis=1)
#     one_hot = np.zeros([pred.shape[0], pred.shape[1]])
#     for hi in range(len(hot)):
#         one_hot[hi, hot[hi]] = 1
#     loss = np.sum(np.absolute(_labels - one_hot)) / 2
#     # print("loss:", loss)
#     #############################################################################################
#     eps = 0
#     while loss > 0 and eps < 200:
#         eps += 1
#         sess.run(train_opt, feed_dict={ph_input: _data, ph_label: _labels})
#         ##############################################################################################
#         pred = sess.run(net_data_output, feed_dict={ph_input: _data, ph_label: _labels})
#         pred = np.array(pred)
#         hot = np.argmax(pred, axis=1)
#         one_hot = np.zeros([pred.shape[0], pred.shape[1]])
#         for hi in range(len(hot)):
#             one_hot[hi, hot[hi]] = 1
#         loss = np.sum(np.absolute(_labels - one_hot)) / 2
#         # print("loss:", loss)
#         ############################################################################################
#     pred = sess.run(net_memory_output, feed_dict={ph_input: _memory_data, ph_label: _memory_labels})
#     pred = np.array(pred)
#     hot = np.argmax(pred, axis=1)
#     one_hot = np.zeros([pred.shape[0], pred.shape[1]])
#     for hi in range(len(hot)):
#         one_hot[hi, hot[hi]] = 1
#     loss = np.sum(np.absolute(_memory_labels - one_hot)) / 2
#     print("memory loss:", loss)
#     tf.reset_default_graph()
#     sess.close()
#     return loss


# def train_new_cell(architecture, ws, bs, ln, data, labels):
#     _data = np.array(copy.deepcopy(data))
#     _labels = np.array(copy.deepcopy(labels))
#     _labels[_labels <= 0] = 0
#     _labels[_labels > 0] = 1
#     sess = tf.InteractiveSession()
#     ph_input = tf.placeholder(dtype=tf.float32, shape=[None, architecture[0]])
#     ph_label = tf.placeholder(dtype=tf.float32, shape=[None, architecture[-1]])
#     net_sign = [ph_input]
#     net_sigmoid = [ph_input]
#     w_list = []
#     b_list = []
#     logits = None
#     for i in range(len(architecture) - 1):
#         if i == ln + 1:
#             new_weight = tf.Variable(tf.random_normal([1, architecture[ln + 2]]), dtype=tf.float32, trainable=True)
#             # new_weight = tf.Variable(_data[], dtype=tf.float32, trainable=True)
#             old_weight = tf.Variable(ws[i], dtype=tf.float32, trainable=False)
#             w_list.append(tf.concat([old_weight, new_weight], axis=0))
#             b_list.append(tf.Variable(bs[i], dtype=tf.float32, trainable=False))
#         else:
#             w_list.append(tf.Variable(ws[i], dtype=tf.float32, trainable=False))
#             b_list.append(tf.Variable(bs[i], dtype=tf.float32, trainable=False))
#         if i == len(architecture) - 2:
#             logits = tf.matmul(net_sigmoid[-1], w_list[-1]) + b_list[-1]
#         net_sigmoid.append(tf.nn.sigmoid(tf.matmul(net_sigmoid[-1], w_list[-1]) + b_list[-1]))
#         net_sign.append(zero_sign(tf.matmul(net_sign[-1], w_list[-1]) + b_list[-1]))
#
#     t_add_f = np.ones(_labels.shape[1]) * len(_data)
#     t_only = np.maximum(np.sum(_labels, axis=0), 0.1)
#     f_div_t = np.divide(t_add_f - t_only, t_only)
#     loss_list = []
#     for i in range(_labels.shape[1]):
#         # loss_list.append(tf.nn.weighted_cross_entropy_with_logits(ph_label[:, i], logits[:, i], f_div_t[i]))
#         loss_list.append(tf.nn.weighted_cross_entropy_with_logits(ph_label[:, i], logits[:, i], 1))
#     loss = tf.reduce_sum(loss_list)
#     train_opt = tf.train.AdamOptimizer(1e-1).minimize(loss)
#     tf.global_variables_initializer().run()
#     ##############################################################################################
#     i = 0
#     count_down = 100
#     min_loss = len(_data)
#     w_out, b_out = None, None
#     while True:
#         sess.run(train_opt, feed_dict={ph_input: _data, ph_label: _labels})
#         ##############################################################################################
#         if i % 50 == 49:
#             # w_zzz, b_zzz = sess.run([w_list[ln + 1], b_list[ln + 1]])
#             # print(w_zzz, b_zzz)
#             pred = sess.run(net_sign[-1], feed_dict={ph_input: _data, ph_label: _labels})
#             hot = np.argmax(pred, axis=1)
#             one_hot = np.zeros([pred.shape[0], pred.shape[1]])
#             for hi in range(len(hot)):
#                 one_hot[hi, hot[hi]] = 1
#             lo = np.sum(np.absolute(_labels - one_hot)) / 2
#             # print("new_cell_loss:", lo)
#             if lo < min_loss:
#                 count_down = 100
#                 min_loss = lo
#                 print("new_cell_loss(min):", min_loss)
#                 w_out, b_out = sess.run([w_list, b_list])
#             else:
#                 count_down -= 1
#             if count_down <= 0:
#                 break
#         i += 1
#     ###########################################################################################
#     tf.reset_default_graph()
#     sess.close()
#     return w_out, b_out


def target_prop(architecture, ws, bs, init_patterns, tar_label, ln, label_shift=None):
    if ln == len(architecture) - 2:
        opt_tar = np.zeros(architecture[-1])
        for i in range(label_shift[0], label_shift[1]):
            opt_tar[i] = tar_label[i - label_shift[0]]
        return opt_tar
    average_pattern = np.mean(init_patterns, axis=0)
    # _data = data
    sess = tf.InteractiveSession()
    # ph_input = tf.placeholder(dtype=tf.float32, shape=[None, architecture[0]])
    net_sign = []
    net_soft = []
    w_list = []
    b_list = []
    target = None
    _label = tf.constant(tar_label, dtype=tf.float32)
    _label_2d = tf.expand_dims(_label, axis=0)
    for i in range(ln, len(architecture) - 1):
        if i == ln:
            target = tf.Variable(average_pattern, dtype=tf.float32)
            target_2d = tf.expand_dims(target, axis=0)
            net_soft.append(zero_softsign(target_2d))
            net_sign.append(zero_sign(target_2d))
            # new_weight = tf.Variable(tf.zeros([1, architecture[ln + 1]]), dtype=tf.float32)
            # old_weight = tf.Variable(ws[i], dtype=tf.float32)
            # w_list.append(tf.concat([old_weight, new_weight], axis=0))
            # b_list.append(tf.Variable(bs[i], dtype=tf.float32))
            # target = tf.matmul(net_sign[-1], w_list[-1]) + b_list[-1]
        elif i < len(architecture) - 2:
            w_list.append(tf.constant(ws[i], dtype=tf.float32))
            b_list.append(tf.constant(bs[i], dtype=tf.float32))
            net_soft.append(zero_softsign(tf.matmul(net_soft[-1], w_list[-1]) + b_list[-1]))
            net_sign.append(zero_sign(tf.matmul(net_sign[-1], w_list[-1]) + b_list[-1]))
        else:
            w_list.append(tf.constant(ws[i], dtype=tf.float32))
            b_list.append(tf.constant(bs[i], dtype=tf.float32))
            net_soft.append(tf.matmul(net_soft[-1], w_list[-1]) + b_list[-1])
            net_sign.append(zero_sign(tf.matmul(net_sign[-1], w_list[-1]) + b_list[-1]))

    if label_shift is not None:
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=net_soft[-1][:, label_shift[0]:label_shift[1]], labels=_label_2d))
    else:
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=net_soft[-1], labels=_label_2d))
    train_opt = tf.train.AdamOptimizer(1e-1).minimize(loss)
    tf.global_variables_initializer().run()
    ##############################################################################################
    for time_cost in range(100):
        for i in range(100):
            sess.run(train_opt)
        lo, opt_tar, sign_tar, pred = sess.run([loss, target, net_sign[0], net_sign[-1]])
        if label_shift is not None:
            pred = pred[:, label_shift[0]:label_shift[1]]
            # opt_tar = opt_tar[label_shift[0]:label_shift[1]]
        if np.sum(np.square(pred - tar_label)) == 0:
            tf.reset_default_graph()
            sess.close()
            return opt_tar
    return None


def max_divide(path_in, path):
    # eps = 80000 # mnist
    eps = 70000
    _path_in = path_in
    _path = path
    sess = tf.InteractiveSession()
    labels = np.concatenate((np.ones([len(_path[0]), 1]), np.zeros([len(_path_in[0]), 1])), axis=0)
    _labels = tf.constant(labels, dtype=tf.float32)
    global_step = tf.Variable(0, trainable=False)
    add_global = global_step.assign_add(1)
    reset_global = global_step.assign(0)
    pos_weight = len(_path_in[0]) / len(_path[0])
    # learning_rate = tf.train.exponential_decay(0.0001, global_step, eps, 1)
    learning_rate = 0.0003
    l_weight = tf.maximum(tf.train.exponential_decay(pos_weight, global_step, int(eps * 0.15), 1 / pos_weight), 1)
    k_list = []
    b_list = []
    pred_list = []
    loss_list = []
    loss_list_g = []
    train_opt_list = []
    train_opt_list_g = []
    for i in range(len(_path) - 1):
        _data = tf.constant(np.concatenate((_path[i], _path_in[i]), axis=0), dtype=tf.float32)
        rint = int(np.random.randint(0, len(_path[i])))
        # if i == 0:
        #     norm = np.sqrt(np.sum(np.square(_path[i][rint]))) / np.sqrt(len(_path[i][rint]))
        #     k = tf.Variable(_path[i][rint:rint+1].T / norm)
        #     # b = tf.Variable([(1e-8 - len(_path[i][rint])) / norm])
        #     max_inner_product = np.max(np.matmul(_path_in[i], (_path[i][rint]) / norm))
        #     inner_product = np.max(np.matmul(_path[i][rint], (_path[i][rint]) / norm))
        #     print(max_inner_product, inner_product, norm)
        #     b = tf.Variable([0.5 * (-max_inner_product - inner_product)])
        # else:
        norm = np.sqrt(np.sum(np.square(_path[i][rint] - 1 / 2))) / np.sqrt(len(_path[i][rint]))
        k = tf.Variable((_path[i][rint:rint + 1].T - 1 / 2) / norm)
        max_inner_product = np.max(np.matmul(_path_in[i] - 1 / 2, (_path[i][rint] - 1 / 2) / norm))
        b = tf.Variable([0.5 * (-max_inner_product - len(_path[i][rint]) * norm - np.sum((_path[i][rint:rint + 1].T - 1 / 2) / norm))])

        k_list.append(k)
        b_list.append(b)
        logits = tf.matmul(_data, k) + b
        # pred = tf.sigmoid(tf.matmul(_data, k) + b)
        pred_list.append(logits)
        loss_w = tf.nn.weighted_cross_entropy_with_logits(targets=_labels, logits=logits, pos_weight=l_weight)
        loss_g = Rel_GW(labels=_labels, logits=logits)
        loss_list.append(tf.reduce_sum(loss_w))
        loss_list_g.append(tf.reduce_sum(loss_g))
        reduce_loss = tf.reduce_sum(loss_list)
        reduce_loss_g = tf.reduce_sum(loss_list_g)
        train_opt = tf.train.AdamOptimizer(learning_rate).minimize(reduce_loss)
        train_opt_g = tf.train.AdamOptimizer(learning_rate).minimize(reduce_loss_g)
        train_opt_list.append(train_opt)
        train_opt_list_g.append(train_opt_g)
    tf.global_variables_initializer().run()

    # temp_test = sess.run(pred_list[0])
    # temp_test0 = temp_test[1:]
    # temp_test1 = temp_test[:2]
    # for i in range(len(temp_test)):
    #     if temp_test[i] > 0:
    #         print(i, temp_test[i])
    #         if i == 0:
    #             print("i==0")
    #         else:
    #             print("i!=0", _path_in[0][i-len(_path[0])])
    # print("temp_test:", temp_test)

    # for li in range(len(_path) - 2):
    for li in range(len(_path) - 1):
        print("layer:", li)
        max_div = 0
        new_list_backup, k_backup, b_backup = None, None, None
        b_sw = 10
        for loop in range(eps):
            if loop % 1000 == 0:
                # print(sess.run([learning_rate, l_weight]))
                pred_v = sess.run(pred_list[li])
                pred_v[pred_v > 0.0] = 1
                pred_v[pred_v <= 0.0] = 0
                t_t = 0
                t_f = 0
                f_t = 0
                f_f = 0
                new_list = []
                for di in range(len(labels)):
                    if pred_v[di] == 1 and labels[di] == 1:
                        t_t += 1
                        new_list.append(di)
                    elif pred_v[di] == 0 and labels[di] == 1:
                        f_t += 1
                    elif pred_v[di] == 1 and labels[di] == 0:
                        t_f += 1
                    elif pred_v[di] == 0 and labels[di] == 0:
                        f_f += 1
                print("new:", t_t, "error:", f_t, "forget:", t_f, "retain:", f_f)
                # #################################################################################################
                if t_t - t_f > max_div and t_t > 1:
                    max_div = t_t - t_f
                    k_backup, b_backup = sess.run([k_list[li], b_list[li]])
                    new_list_backup = new_list
                if f_t == 0 and t_f == 0:
                    break
                if t_t == 0 and t_f == 0:
                    b_sw -= 1
                    print("break_count_down:", b_sw)
                    if b_sw == 0:
                        break
                else:
                    b_sw = 10
                # if sess.run(l_weight) * t_t <= t_f and loop >= 5000:
                #     break
                # #################################################################################################
            # if loop <= 10000:
            #     sess.run(train_opt_list_g[li])
            # else:
            #     sess.run(train_opt_list[li])
            sess.run(train_opt_list_g[li])
            sess.run(add_global)
        sess.run(reset_global)
        pred_v = sess.run(pred_list[li])
        pred_v[pred_v > 0.0] = 1
        pred_v[pred_v <= 0.0] = 0
        t_t = 0
        t_f = 0
        f_t = 0
        f_f = 0
        new_list = []
        for di in range(len(labels)):
            if pred_v[di] == 1 and labels[di] == 1:
                t_t += 1
                new_list.append(di)
            elif pred_v[di] == 0 and labels[di] == 1:
                f_t += 1
            elif pred_v[di] == 1 and labels[di] == 0:
                t_f += 1
            elif pred_v[di] == 0 and labels[di] == 0:
                f_f += 1
        print("new:", t_t, "error:", f_t, "forget:", t_f, "retain:", f_f)
        if t_t - t_f > max_div and t_t > 1 or f_t == 0 and t_f == 0:
            max_div = t_t - t_f
            k_backup, b_backup = sess.run([k_list[li], b_list[li]])
            new_list_backup = new_list
        if max_div > 0:
            best_layer = li
            print("target_layer", best_layer)
            tf.reset_default_graph()
            sess.close()
            return best_layer, new_list_backup, k_backup, b_backup
    tf.reset_default_graph()
    sess.close()
    return None, None, None, None


# def layer_max_divide(path_in, path, ln):
#     _path_in = copy.deepcopy(path_in)
#     _path = copy.deepcopy(path)
#     sess = tf.InteractiveSession()
#     labels = np.concatenate((np.ones([len(_path[0]), 1]), np.zeros([len(_path_in[0]), 1])), axis=0)
#     _labels = tf.constant(labels, dtype=tf.float32)
#     global_step = tf.Variable(0, trainable=False)
#     add_global = global_step.assign_add(1)
#     # reset_global = global_step.assign(0)
#     pos_weight = len(_path_in[0]) / len(_path[0])
#     learning_rate = tf.train.exponential_decay(0.01, global_step, 200000, 1)
#     l_weight = tf.maximum(tf.train.exponential_decay(pos_weight, global_step, 190000, 1 / pos_weight), 1)
#     # k_list = []
#     # b_list = []
#     # pred_list = []
#     # loss_list = []
#     # train_opt_list = []
#     _data = tf.constant(np.concatenate((_path[ln], _path_in[ln]), axis=0), dtype=tf.float32)
#     ##############################################################################################################
#     rint = int(np.random.randint(0, len(_path[ln])))
#     # if ln == 0:
#     #     norm = np.sqrt(np.sum(np.square(_path[ln][rint]))) / np.sqrt(len(_path[ln][rint]))
#     #     k = tf.Variable(_path[ln][rint:rint + 1].T / norm)
#     #     m_sum = np.sum(np.square(_path[ln][rint:rint + 1]) / norm)
#     #     b = tf.Variable([0.000001 - m_sum])
#     # else:
#     norm = np.sqrt(np.sum(np.square(_path[ln][rint] - 1 / 2))) / np.sqrt(len(_path[ln][rint]))
#     k = tf.Variable((_path[ln][rint:rint + 1].T - 1 / 2) / norm)
#     m_sum = np.sum(np.matmul(_path[ln][rint:rint + 1], (_path[ln][rint:rint + 1].T - 1 / 2) / norm))
#     b = tf.Variable([0.00001 - m_sum])
#     #############################################################################################################
#     # k = tf.Variable(tf.random.normal([len(_path[ln][0]), 1], stddev=0.01))
#     # b = tf.Variable(tf.random.normal([1], stddev=0.01))
#     ############################################################################################################
#     logits = tf.matmul(_data, k) + b
#     # pred = tf.sigmoid(tf.matmul(_data, k) + b)
#     # pred_list.append(logits)
#     loss = tf.reduce_sum(tf.nn.weighted_cross_entropy_with_logits(targets=_labels, logits=logits, pos_weight=l_weight))
#     # loss_list.append(tf.reduce_sum(loss))
#     # reduce_loss = tf.reduce_sum(loss_list)
#     train_opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)
#     # train_opt_list.append(train_opt)
#     tf.global_variables_initializer().run()
#     max_div = 0
#     new_list_backup, k_backup, b_backup = None, None, None
#     for loop in range(200000):
#         if loop % 1000 == 0:
#             print(sess.run([learning_rate, l_weight]))
#             pred_v = sess.run(logits)
#             pred_v[pred_v > 0.0] = 1
#             pred_v[pred_v <= 0.0] = 0
#             t_t = 0
#             t_f = 0
#             f_t = 0
#             f_f = 0
#             new_list = []
#             for di in range(len(labels)):
#                 if pred_v[di] == 1 and labels[di] == 1:
#                     t_t += 1
#                     new_list.append(di)
#                 elif pred_v[di] == 0 and labels[di] == 1:
#                     f_t += 1
#                 elif pred_v[di] == 1 and labels[di] == 0:
#                     t_f += 1
#                 elif pred_v[di] == 0 and labels[di] == 0:
#                     f_f += 1
#             print("new:", t_t, "error:", f_t, "forget:", t_f, "remember:", f_f)
#             #################################################################################################
#             if t_t - t_f > max_div:
#                 max_div = t_t - t_f
#                 k_backup, b_backup = sess.run([k, b])
#                 new_list_backup = copy.deepcopy(new_list)
#             #################################################################################################
#         sess.run(train_opt)
#         sess.run(add_global)
#     # sess.run(reset_global)
#     pred_v = sess.run(logits)
#     pred_v[pred_v > 0.0] = 1
#     pred_v[pred_v <= 0.0] = 0
#     t_t = 0
#     t_f = 0
#     f_t = 0
#     f_f = 0
#     new_list = []
#     for di in range(len(labels)):
#         if pred_v[di] == 1 and labels[di] == 1:
#             t_t += 1
#             new_list.append(di)
#         elif pred_v[di] == 0 and labels[di] == 1:
#             f_t += 1
#         elif pred_v[di] == 1 and labels[di] == 0:
#             t_f += 1
#         elif pred_v[di] == 0 and labels[di] == 0:
#             f_f += 1
#     print("new:", t_t, "error:", f_t, "forget:", t_f, "remember:", f_f)
#     if t_t - t_f > max_div:
#         max_div = t_t - t_f
#         k_backup, b_backup = sess.run([k, b])
#         new_list_backup = copy.deepcopy(new_list)
#     print("max_div:", max_div)
#     tf.reset_default_graph()
#     sess.close()
#     return new_list_backup, k_backup, b_backup


def train_next_layer(architecture, ws, bs, ln, data, labels, mask=None):
    _data = data
    _labels = labels
    sess = tf.InteractiveSession()
    ph_input = tf.placeholder(dtype=tf.float32, shape=[None, architecture[0]])
    new_weight = tf.placeholder(dtype=tf.float32, shape=[1, architecture[ln + 2]])
    # ph_label = tf.placeholder(dtype=tf.float32, shape=[None, architecture[-1]])
    net_sign = [ph_input]
    w_list = []
    b_list = []
    reduce_max, reduce_min = None, None
    for i in range(len(architecture) - 1):
        if i == ln + 1:
            # new_weight = tf.Variable(tf.zeros([1, architecture[ln + 2]]), dtype=tf.float32, trainable=False)
            old_weight = tf.Variable(ws[i], dtype=tf.float32, trainable=False)
            w_list.append(tf.concat([old_weight, new_weight], axis=0))
            b_list.append(tf.Variable(bs[i], dtype=tf.float32, trainable=False))
            value = tf.matmul(net_sign[-1], w_list[-1]) + b_list[-1]
            reduce_max = tf.reduce_max(value, axis=0)
            reduce_min = tf.reduce_min(value, axis=0)
            net_sign.append(zero_sign(tf.matmul(net_sign[-1], w_list[-1]) + b_list[-1]))
        else:
            w_list.append(tf.Variable(ws[i], dtype=tf.float32, trainable=False))
            b_list.append(tf.Variable(bs[i], dtype=tf.float32, trainable=False))
            net_sign.append(zero_sign(tf.matmul(net_sign[-1], w_list[-1]) + b_list[-1]))

    tf.global_variables_initializer().run()
    ##############################################################################################
    # val = sess.run(value, feed_dict={ph_input: _data})
    # print(val)
    max_array, min_array = sess.run([reduce_max, reduce_min], feed_dict={ph_input: _data, new_weight: np.zeros([1, architecture[ln + 2]])})
    # print("max, min:", max_array, min_array)
    new_w = []
    shift = 1
    for ti in range(architecture[ln + 2]):
        if mask is not None:
            if ti < mask[ln + 1]:
                new_w.append(0)
            else:
                if _labels[ti] > 0:
                    new_w.append(np.maximum(shift - min_array[ti], shift))
                else:
                    new_w.append(np.minimum(-shift - max_array[ti], -shift))
        else:
            if _labels[ti] > 0:
                new_w.append(np.maximum(shift - min_array[ti], shift))
            else:
                new_w.append(np.minimum(-shift - max_array[ti], -shift))

    w_out, b_out = sess.run([w_list, b_list], feed_dict={new_weight: [new_w]})
    print("new weight:", w_out[ln + 1][-1])

    ###########################################################################################
    tf.reset_default_graph()
    sess.close()
    return w_out, b_out


def train_new_parameters(architecture, ws, bs, ln, data, labels, mask=None):
    _data = data
    _labels = np.array(labels, dtype=np.float32)
    # for i in range(len(_data)):
    #     _labels = _labels + [labels]
    # print(len(_labels), len(_labels[0]))
    sess = tf.InteractiveSession()
    net_sigmoid = [_data]
    net_sign = [_data]
    w_list = []
    b_list = []
    for i in range(len(architecture) - 1):
        if i == ln + 1:
            if mask is not None:
                new_weight_u = tf.Variable(tf.zeros([1, mask[ln + 1]]), dtype=tf.float32, trainable=True)
                new_weight_t = tf.Variable(tf.random.normal([1, architecture[ln + 2] - mask[ln + 1]], stddev=1e-2),
                                           dtype=tf.float32, trainable=True)
                new_weight = tf.concat([new_weight_u, new_weight_t], axis=1)
            else:
                new_weight = tf.Variable(tf.zeros([1, architecture[ln + 2]]), dtype=tf.float32, trainable=True)
            old_weight = tf.Variable(ws[i], dtype=tf.float32, trainable=False)
            w_list.append(tf.concat([old_weight, new_weight], axis=0))
            b_list.append(tf.Variable(bs[i], dtype=tf.float32, trainable=False))
        else:
            w_list.append(tf.Variable(ws[i], dtype=tf.float32, trainable=False))
            b_list.append(tf.Variable(bs[i], dtype=tf.float32, trainable=False))
        if i == len(architecture) - 2:
            if mask is not None:
                logits = tf.add(tf.matmul(net_sigmoid[-1], w_list[-1]), b_list[-1])[:, mask[-1]:]
            else:
                logits = tf.add(tf.matmul(net_sigmoid[-1], w_list[-1]), b_list[-1])
        net_sigmoid.append(zero_softsign(tf.matmul(net_sigmoid[-1], w_list[-1]) + b_list[-1]))
        net_sign.append(zero_sign(tf.matmul(net_sign[-1], w_list[-1]) + b_list[-1]))
    if mask is not None:
        acc = 1.0 - tf.reduce_sum(tf.square(net_sign[-1][:, mask[-1]:] - _labels)) / (2 * len(_labels))
    else:
        acc = 1.0 - tf.reduce_sum(tf.square(net_sign[-1] - _labels)) / (2 * len(_labels))
    loss_list = []
    for i in range(len(_labels[0])):
        # loss_list.append(tf.nn.weighted_cross_entropy_with_logits(ph_label[:, i], logits[:, i], ph_weighted[i]))
        loss_list.append(Rel_GW(_labels[:, i:i+1], logits[:, i:i+1]))
    loss = tf.reduce_mean(loss_list)
    train_opt = tf.train.AdamOptimizer(1e-2).minimize(loss)

    tf.global_variables_initializer().run()
    ##############################################################################################
    # val = sess.run(value, feed_dict={ph_input: _data})
    # print(val)
    accuracy, nw = sess.run([acc, new_weight])
    print(-1, accuracy, nw)
    for i in range(1000):
        _, accuracy, nw = sess.run([train_opt, acc, new_weight])
        if i % 10 == 0:
            print(i, accuracy, nw)

    w_out, b_out = sess.run([w_list, b_list])
    print("new weight:", w_out[ln + 1][-1])

    ###########################################################################################
    tf.reset_default_graph()
    sess.close()
    return w_out, b_out
