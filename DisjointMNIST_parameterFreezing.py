import SGBAN
import numpy as np
import copy


# data preparing
mnist0to4 = np.load('DisjointMNIST/mnist0to4.npz')
mnist5to9 = np.load('DisjointMNIST/mnist5to9.npz')
train_set = mnist5to9["train_data"].tolist()
train_set_labels = mnist5to9["train_labels"].tolist()
test_set_1 = mnist0to4["test_data"].tolist()
test_set_labels_1 = mnist0to4["test_labels"].tolist()
test_set_2 = mnist5to9["test_data"].tolist()
test_set_labels_2 = mnist5to9["test_labels"].tolist()


# network restore
net = SGBAN.Network(784, 5, 'Model/save_disjointMNIST.json')
net.load()


# preparing for task 2
net.mask = copy.deepcopy(net.architecture[1:])
for i in range(5):
    net.add_cell(len(net.architecture) - 2)
for i in range(0, len(net.architecture) - 2):
    for j in range(5):
        net.add_cell(i)
        m = 0
        for ci in net.layers[i + 1].neurons:
            if m < net.mask[i + 1]:
                ci.k = ci.k + [[0]]
            else:
                ci.k = ci.k + [[np.random.normal()]]
            m += 1
net.save("task2")
print("task 1 accuracy:")
net.test(test_set_1, test_set_labels_1, softmax=True, label_shift=[0, 5])
print("task 2 accuracy:")
net.test(test_set_2, test_set_labels_2, softmax=True, label_shift=[5, 10])


# warm-up
for i in range(20):
    if i < 5:
        net.fine_tune(train_set, train_set_labels, 1000, 1000, lr=1e-2)
    else:
        net.fine_tune(train_set, train_set_labels, 1000, 1000, lr=1e-3)
print("task 2 accuracy:")
net.test(test_set_2, test_set_labels_2, softmax=True, label_shift=[5, 10])
net.save("task2")


# training
net.in_memory_data = copy.deepcopy(train_set)
net.in_memory_labels = copy.deepcopy(train_set_labels)
net.out_memory_data = []
net.out_memory_labels = []
net.check_memory()
i = 0
l_num = len(net.architecture) - 1
while len(net.out_memory_data) > 0:
    net.training(i)
    print("loop", i, "architecture", net.architecture)
    i += 1
    if l_num < len(net.architecture) - 1:
        l_num += 1
        for j in range(3):
            net.try_fine_tuning(loop=10000)
    print("task 2 accuracy:")
    net.test(test_set_2, test_set_labels_2, softmax=True, label_shift=[5, 10])
    if i == 30:
        for j in range(30):
            if j < 5:
                net.fine_tune(train_set, train_set_labels, 1000, 1000, lr=1e-1)
            else:
                net.in_memory_data = copy.deepcopy(train_set)
                net.in_memory_labels = copy.deepcopy(train_set_labels)
                net.out_memory_data = []
                net.out_memory_labels = []
                net.check_memory()
                net.fine_tune(net.in_memory_data, net.in_memory_labels, 1000, 1000, lr=1e-3)
net.save("task2")


# fine-tuning
for i in range(30):
    if i < 5:
        net.fine_tune(train_set, train_set_labels, 1000, 1000, lr=1e-1)
    else:
        net.in_memory_data = copy.deepcopy(train_set)
        net.in_memory_labels = copy.deepcopy(train_set_labels)
        net.out_memory_data = []
        net.out_memory_labels = []
        net.check_memory()
        net.fine_tune(net.in_memory_data, net.in_memory_labels, 1000, 1000, lr=1e-3)
print("task 1 accuracy:")
net.test(test_set_1, test_set_labels_1, softmax=True, label_shift=[0, 5])
print("task 2 accuracy:")
net.test(test_set_2, test_set_labels_2, softmax=True, label_shift=[5, 10])
