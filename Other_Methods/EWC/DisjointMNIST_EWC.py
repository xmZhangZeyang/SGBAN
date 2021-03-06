from EWC import ewc_network as ewc
import numpy as np


mnist0to4 = np.load('../DisjointMNIST/mnist0to4.npz')
mnist5to9 = np.load('../DisjointMNIST/mnist5to9.npz')

train_set_59 = mnist5to9["train_data"].tolist()
train_set_labels_59 = mnist5to9["train_labels"].tolist()
test_set_59 = mnist5to9["test_data"].tolist()
test_set_labels_59 = mnist5to9["test_labels"].tolist()

train_set_04 = mnist0to4["train_data"].tolist()
train_set_labels_04 = mnist0to4["train_labels"].tolist()
test_set_04 = mnist0to4["test_data"].tolist()
test_set_labels_04 = mnist0to4["test_labels"].tolist()


model_path = "../Models/model_ewc.npz"
batch_size = 10000
ld = 1e4
architecture = [784, 79, 11, 5]
net = ewc.Network(architecture, ld, model=None)


#  train task 1
net.clear_fisher()
for i in range(100):
    net.train(train_set_04, train_set_labels_04, 100, batch_size, 1e-2)
    net.eval(test_set_04, test_set_labels_04)


net.set_star()
net.update_fisher(train_set_04, 1000)


#  train task 2
for i in range(500):
    net.train(train_set_59, train_set_labels_59, 100, batch_size, 1e-4)
    print("task1:")
    net.eval(test_set_04, test_set_labels_04)
    print("task2:")
    net.eval(test_set_59, test_set_labels_59)

net.save(model_path)
net.close()
