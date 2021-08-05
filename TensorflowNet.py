import tensorflow as tf
import numpy as np
import os


os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'


def zero_sign(tensor):
    return tf.maximum(tf.sign(tensor), 0)


def zero_softsign(tensor):
    return (tf.nn.tanh(tensor) + 1) / 2


def Rel_GW(labels, logits):
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


def fine_tune(architecture, ws, bs, data, labels, batch_size=1000, loop=100, lr=1e-6, mask=None):
    if mask is not None:
        print("mask:", mask)
    _data = np.array(data)
    _labels = np.array(labels)
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
    loss_list = []
    for i in range(len(_labels[0])):
        loss_list.append(tf.nn.weighted_cross_entropy_with_logits(ph_label[:, i], logits[:, i], ph_weighted[i]))
    loss = tf.reduce_mean(loss_list)
    train_opt = tf.train.AdamOptimizer(lr).minimize(loss)
    tf.global_variables_initializer().run()
    eps = 0
    while eps < loop:
        random = np.random.permutation(len(data))
        random_data_all = _data[random]
        random_labels_all = _labels[random]
        if len(data) < batch_size:
            random_data = random_data_all
            random_labels = random_labels_all
            eps += 1
            f_div_t = np.ones(_labels.shape[1])
            sess.run(train_opt, feed_dict={ph_input: random_data, ph_label: random_labels, ph_weighted: f_div_t})
        else:
            for bi in range(len(data) // batch_size):
                random_data = random_data_all[bi * batch_size:(bi+1) * batch_size]
                random_labels = random_labels_all[bi * batch_size:(bi+1) * batch_size]
                eps += 1
                f_div_t = np.ones(_labels.shape[1])
                sess.run(train_opt, feed_dict={ph_input: random_data, ph_label: random_labels, ph_weighted: f_div_t})
    w_new, b_new = sess.run([w_list, b_list])
    tf.reset_default_graph()
    sess.close()
    return w_new, b_new


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


def target_prop(architecture, ws, bs, init_patterns, tar_label, ln, mask=None, label_shift=None, threshold=0):
    average_pattern = np.mean(init_patterns, axis=0).tolist()
    sess = tf.InteractiveSession()
    net_sign = []
    net_soft = []
    w_list = []
    b_list = []
    target_2d = None
    _label = tf.constant([tar_label], dtype=tf.float32)
    _label_2d = _label
    for i in range(ln, len(architecture) - 1):
        if i == ln:
            if mask is not None:
                target_u = tf.constant(init_patterns[:, :mask[ln]], dtype=tf.float32)
                target = tf.Variable([average_pattern[mask[ln]:]], dtype=tf.float32)
                target_2d = target
                for pi in range(len(init_patterns) - 1):
                    target_2d = tf.concat([target_2d, target], axis=0)
                    _label_2d = tf.concat([_label_2d, _label], axis=0)
                target_2d = tf.concat([target_u, target_2d], axis=1)
                net_soft.append(zero_softsign(target_2d))
                net_sign.append(zero_sign(target_2d))
            else:
                target = tf.Variable(average_pattern, dtype=tf.float32)
                target_2d = tf.expand_dims(target, axis=0)
                net_soft.append(zero_softsign(target_2d))
                net_sign.append(zero_sign(target_2d))
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
    train_opt = tf.train.AdamOptimizer(1e-2).minimize(loss)
    tf.global_variables_initializer().run()
    max_count = 0
    opt_tar_backup = None
    for time_cost in range(10):
        for i in range(100):
            sess.run(train_opt)
        lo, opt_tar, sign_tar, pred = sess.run([loss, target_2d[0], net_sign[0], net_sign[-1]])
        if label_shift is not None:
            pred = pred[:, label_shift[0]:label_shift[1]]
        if mask is not None:
            count = 0
            for i in range(len(init_patterns)):
                if np.sum(np.square(pred[i] - tar_label)) == 0:
                    count += 1
            if count > threshold and count > max_count:
                max_count = count
                opt_tar_backup = opt_tar
                print("target_prop:", count, "/", len(init_patterns))
        else:
            if np.sum(np.square(pred - tar_label)) == 0:
                tf.reset_default_graph()
                sess.close()
                return opt_tar
    tf.reset_default_graph()
    sess.close()
    return opt_tar_backup


def max_divide(path_in, path):
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
        norm = np.sqrt(np.sum(np.square(_path[i][rint] - 1 / 2))) / np.sqrt(len(_path[i][rint]))
        k = tf.Variable((_path[i][rint:rint + 1].T - 1 / 2) / norm)
        max_inner_product = np.max(np.matmul(_path_in[i] - 1 / 2, (_path[i][rint] - 1 / 2) / norm))
        b = tf.Variable([0.5 * (-max_inner_product - len(_path[i][rint]) * norm - np.sum((_path[i][rint:rint + 1].T - 1 / 2) / norm))])

        k_list.append(k)
        b_list.append(b)
        logits = tf.matmul(_data, k) + b
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

    for li in range(len(_path) - 1):
        print("layer:", li)
        max_div = 0
        new_list_backup, k_backup, b_backup, t_f_backup = None, None, None, None
        b_sw = 10
        for loop in range(eps):
            if loop % 1000 == 0:
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
                if t_t - t_f > max_div and t_t > 1:
                    print("new:", t_t, "error:", f_t, "forget:", t_f, "retain:", f_f)
                    max_div = t_t - t_f
                    k_backup, b_backup = sess.run([k_list[li], b_list[li]])
                    new_list_backup = new_list
                    t_f_backup = t_f
                if f_t == 0 and t_f == 0:
                    break
                if t_t == 0 and t_f == 0:
                    b_sw -= 1
                    if b_sw == 0:
                        break
                else:
                    b_sw = 10
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
        if t_t - t_f > max_div and t_t > 1 or f_t == 0 and t_f == 0:
            print("new:", t_t, "error:", f_t, "forget:", t_f, "retain:", f_f)
            max_div = t_t - t_f
            k_backup, b_backup = sess.run([k_list[li], b_list[li]])
            new_list_backup = new_list
            t_f_backup = t_f
        if max_div > 0:
            best_layer = li
            print("target_layer", best_layer)
            tf.reset_default_graph()
            sess.close()
            return best_layer, new_list_backup, k_backup, b_backup, t_f_backup
    tf.reset_default_graph()
    sess.close()
    return None, None, None, None, None


def train_next_layer(architecture, ws, bs, ln, data, labels, mask=None):
    _data = data
    _labels = labels
    sess = tf.InteractiveSession()
    ph_input = tf.placeholder(dtype=tf.float32, shape=[None, architecture[0]])
    new_weight = tf.placeholder(dtype=tf.float32, shape=[1, architecture[ln + 2]])
    net_sign = [ph_input]
    w_list = []
    b_list = []
    reduce_max, reduce_min = None, None
    for i in range(len(architecture) - 1):
        if i == ln + 1:
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
    max_array, min_array = sess.run([reduce_max, reduce_min],
                                    feed_dict={ph_input: _data, new_weight: np.zeros([1, architecture[ln + 2]])})
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
    print("new weights of next layer:", w_out[ln + 1][-1])
    tf.reset_default_graph()
    sess.close()
    return w_out, b_out


def train_new_parameters(architecture, ws, bs, ln, data, labels, mask=None):
    _data = data
    _labels = np.array(labels, dtype=np.float32)
    sess = tf.InteractiveSession()
    net_sigmoid = [_data]
    net_sign = [_data]
    w_list = []
    b_list = []
    for i in range(len(architecture) - 1):
        if i == ln + 1:
            if mask is not None:
                new_weight_u = tf.Variable(tf.zeros([1, mask[ln + 1]]), dtype=tf.float32, trainable=False)
                new_weight_t = tf.Variable(tf.random.normal([1, architecture[ln + 2] - mask[ln + 1]], stddev=1.0),
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
    loss_list = []
    for i in range(len(_labels[0])):
        loss_list.append(Rel_GW(_labels[:, i:i+1], logits[:, i:i+1]))
    loss = tf.reduce_mean(loss_list)
    train_opt = tf.train.AdamOptimizer(1e-2).minimize(loss)
    tf.global_variables_initializer().run()
    for i in range(1000):
        sess.run(train_opt)
    w_out, b_out = sess.run([w_list, b_list])
    print("new weights of next layer:", w_out[ln + 1][-1])
    tf.reset_default_graph()
    sess.close()
    return w_out, b_out
