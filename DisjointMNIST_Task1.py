import tensorflow.examples.tutorials.mnist.input_data as input_data
import SGBAN
import numpy as np
import copy


# data preparing
mnist0to4 = np.load('DisjointMNIST/mnist0to4.npz')

train_set = mnist0to4["train_data"].tolist()
train_set_labels = mnist0to4["train_labels"].tolist()
test_set = mnist0to4["test_data"].tolist()
test_set_labels = mnist0to4["test_labels"].tolist()


# network initialization
net = SGBAN.Network(784, 5, 'models/save_disjointMNIST.json')

# for i in range(30):
#     if i < 5:
#         net.fine_tune(train_set, train_set_labels, 1000, 1000, lr=1e-2)
#     else:
#         net.fine_tune(train_set, train_set_labels, 1000, 1000, lr=1e-3)
#
# net.in_memory_data = copy.deepcopy(train_set)
# net.in_memory_labels = copy.deepcopy(train_set_labels)
# net.out_memory_data = []
# net.out_memory_labels = []
# net.check_memory()
# net.save()
net.load()
net.test(test_set, test_set_labels, softmax=True)
exit()


# training
i = 0
l_num = len(net.architecture) - 1
while len(net.out_memory_data) > 0:
    net.training()
    print("loop", i, "architecture", net.architecture)
    i += 1
    if l_num < len(net.architecture) - 1:
        l_num += 1
        for i in range(3):
            net.try_fine_tuning(loop=10000)
    net.test(test_set, test_set_labels, softmax=True)
    net.save()
