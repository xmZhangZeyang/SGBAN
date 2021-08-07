import numpy as np
import fcn
from cvxopt import matrix, solvers


def get_batch(data, labels, batchsize, is_test_set=False):
    if is_test_set is False:
        rand_i = np.random.permutation(len(data))
    else:
        rand_i = range(len(data))
    rand_data = []
    rand_labels = []
    shift = 0
    for i in rand_i:
        rand_data.append(data[i])
        rand_labels.append(labels[i])
        shift += 1
        if shift % batchsize == 0 or shift >= len(data):
            yield rand_data, rand_labels
            rand_data = []
            rand_labels = []


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


net_arc = [784, 79, 11, 5]
net = fcn.Fully_Connected_Network(net_arc, ["sigmoid", "sigmoid", "softmax"])
net.restore('Models/model_gem/')

# hyper_parameters
lr = 1e-3
batch_size = 1000
print("learning_rate =", lr, "    batch_size =", batch_size)

# before training
test_batch_generator = get_batch(test_set_04, test_set_labels_04, batch_size, is_test_set=True)
sum_lo = 0
sum_acc = 0
for (data, labels) in test_batch_generator:
    _, correct_soft, correct_sign = net.prediction(data, labels)
    sum_acc += correct_soft
test_acc1 = sum_acc / len(test_set_labels_04)
best_acc = test_acc1
print("before t1, test_t1_acc:", test_acc1)

# training task 1
for ei in range(0, 300):
    train_batch_generator = get_batch(train_set_04, train_set_labels_04, batch_size)
    for (data, labels) in train_batch_generator:
        net.training(data, labels, lr)
    # eval
    test_batch_generator = get_batch(test_set_04, test_set_labels_04, batch_size, is_test_set=True)
    sum_lo = 0
    sum_acc = 0
    for (data, labels) in test_batch_generator:
        _, correct_soft, correct_sign = net.prediction(data, labels)
        sum_acc += correct_soft
        lo = net.get_loss(data, labels)
        sum_lo += lo
    test_acc = sum_acc / len(test_set_labels_04)
    test_lo = sum_lo / len(test_set_labels_04)
    print(ei, "test_acc:", test_acc, "test_loss:", test_lo)
    if test_acc > best_acc:
        best_acc = test_acc
        net.save('Models/model_gem/')
        print("saved.")
exit()


# training task 2
lr = 1e-3
batch_size = 1000
test_batch_generator = get_batch(test_set_59, test_set_labels_59, batch_size, is_test_set=True)
sum_lo = 0
sum_acc = 0
for (data, labels) in test_batch_generator:
    _, correct_soft, correct_sign = net.prediction(data, labels)
    sum_acc += correct_soft
test_acc2 = sum_acc / len(test_set_labels_59)
best_acc = test_acc1 * 0.5139 + test_acc2 * 0.4861
print("before t2, test_t2_acc:", test_acc2)
print("best_average_acc:", best_acc)


for ei in range(0, 1000):
    train_batch_generator = get_batch(train_set_59, train_set_labels_59, batch_size)
    for (data, labels) in train_batch_generator:
        # old tasks gradients
        grad_list = net.get_grads(train_set_04[:2570], train_set_labels_04[:2570])
        _g = np.array([])
        for i in range(len(grad_list)):
            g_shape = np.shape(grad_list[i])
            dim = 1
            for di in g_shape:
                dim *= di
            _g = np.concatenate([_g, np.reshape(grad_list[i], dim)], axis=0)
        G = - np.expand_dims(_g, axis=0)

        # new task gradients
        grad_list = net.get_grads(data, labels)
        g_new_task_only = np.array([])
        for i in range(len(grad_list)):
            g_shape = np.shape(grad_list[i])
            dim = 1
            for di in g_shape:
                dim *= di
            g_new_task_only = np.concatenate([g_new_task_only, np.reshape(grad_list[i], dim)], axis=0)
        g_t = - np.expand_dims(g_new_task_only, axis=0)
        # qp
        qp_P = matrix(np.matmul(G, np.transpose(G)).tolist())
        qp_q = matrix(np.matmul(g_t, np.transpose(G))[0].tolist())
        qp_G = matrix([[-1.0]])
        qp_h = matrix([0.0])
        result = solvers.qp(qp_P, qp_q, G=qp_G, h=qp_h, solvers='mosek')
        v = result['x'][0] + 1e-3
        grad_for_update = _g * v + g_new_task_only
        inner_product = np.matmul(grad_for_update, np.transpose(_g))
        print('v:', result['x'][0], v, "g*G:", qp_q, "inner product:", inner_product)
        if inner_product < 0:
            print("error")
        else:
            net.apply_grads(grad_for_update, lr)
    # eval task1
    test_batch_generator = get_batch(test_set_04, test_set_labels_04, batch_size, is_test_set=True)
    sum_lo = 0
    sum_acc = 0
    for (data, labels) in test_batch_generator:
        _, correct_soft, correct_sign = net.prediction(data, labels)
        sum_acc += correct_soft
        lo = net.get_loss(data, labels)
        sum_lo += lo
    test_acc1 = sum_acc / len(test_set_labels_04)
    test_lo1 = sum_lo / len(test_set_labels_04)
    print(ei, "test_t1_acc:", test_acc1, "test_t1_loss:", test_lo1)
    # eval task2
    test_batch_generator = get_batch(test_set_59, test_set_labels_59, batch_size, is_test_set=True)
    sum_lo = 0
    sum_acc = 0
    for (data, labels) in test_batch_generator:
        _, correct_soft, correct_sign = net.prediction(data, labels)
        sum_acc += correct_soft
        lo = net.get_loss(data, labels)
        sum_lo += lo
    test_acc2 = sum_acc / len(test_set_labels_59)
    test_lo2 = sum_lo / len(test_set_labels_59)
    print(ei, "test_t2_acc:", test_acc2, "test_t2_loss:", test_lo2)
    # save
    avg_acc = test_acc1 * 0.5139 + test_acc2 * 0.4861
    if avg_acc > best_acc:
        best_acc = avg_acc
        print("best_average_acc:", best_acc)
        net.save('Models/model_gem/')
        print("saved.")

