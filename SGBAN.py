import numpy as np
import json
# import copy
# import math
# from numba import cuda, jit
import Operation as Opt
# import TensorflowNet_data_expand as Tn
import TensorflowNet as Tn
import time


# def set_cleaning(table):
#     if len(table) == 0:
#         return
#     else:
#         length = len(table) - 1
#         i = 0
#         while i < length:
#             if min_distance(table[0:i + 1], table[i + 1]) == 0:
#                 del table[i + 1]
#                 length -= 1
#             else:
#                 i += 1
#
#
# def mutual_information(patterns1, patterns2):
#     if len(patterns1) != len(patterns2):
#         print("ERROR: mutual_information")
#         exit()
#     p1 = np.array(patterns1)
#     p2 = np.array(patterns2)
#     p_arr = np.zeros([len(patterns2), len(patterns1)])
#     sum_all = len(patterns1)
#     for n in range(len(patterns1)):
#         p_arr[n, n] = 1
#     for i in range(p_arr.shape[0]):
#         if i < p_arr.shape[0]:
#             i_eq = np.sum(np.square(p2[i + 1:] - p2[i]), axis=1)
#             del_list = []
#             for j in range(len(i_eq)):
#                 if i_eq[j] == 0:
#                     p_arr[i] += p_arr[i + 1 + j]
#                     del_list.append(i + 1 + j)
#             p_arr = np.delete(p_arr, del_list, axis=0)
#             p2 = np.delete(p2, del_list, axis=0)
#     p_arr = p_arr.T
#
#     for i in range(p_arr.shape[0]):
#         if i < p_arr.shape[0]:
#             i_eq = np.sum(np.square(p1[i + 1:] - p1[i]), axis=1)
#             del_list = []
#             for j in range(len(i_eq)):
#                 if i_eq[j] == 0:
#                     p_arr[i] += p_arr[i + 1 + j]
#                     del_list.append(i + 1 + j)
#             p_arr = np.delete(p_arr, del_list, axis=0)
#             p1 = np.delete(p1, del_list, axis=0)
#     print("I_array:", p_arr.shape, np.sum(p_arr))
#     sum1 = np.sum(p_arr, axis=1)
#     sum2 = np.sum(p_arr, axis=0)
#     h1 = 0
#     h2 = 0
#     for si in sum1:
#         h1 += entropy(si / sum_all)
#     for si in sum2:
#         h2 += entropy(si / sum_all)
#     print("H:", h1, h2)
#     print("H / log(q):", h1 / np.log2(p_arr.shape[0]), h2 / np.log2(p_arr.shape[1]))
#     mi = 0
#     for i in range(p_arr.shape[0]):
#         for j in range(p_arr.shape[1]):
#             if p_arr[i, j] != 0:
#                 mi += p_arr[i, j] / sum_all * np.log2(p_arr[i, j] * sum_all / (sum1[i] * sum2[j]))
#     print("I:", mi)
#     return mi / h1
#
#
# def entropy_rr(patterns):
#     _p = np.array(patterns)
#     p_sum = _p.shape[0]
#     p_arr = np.ones(_p.shape[0])
#     for i in range(_p.shape[0]):
#         if i < _p.shape[0]:
#             i_eq = np.sum(np.square(_p[i + 1:] - _p[i]), axis=1)
#             del_list = []
#             for j in range(len(i_eq)):
#                 if i_eq[j] == 0:
#                     p_arr[i] += p_arr[i + 1 + j]
#                     del_list.append(i + 1 + j)
#             p_arr = np.delete(p_arr, del_list)
#             _p = np.delete(_p, del_list, axis=0)
#     p_arr = p_arr.T
#
#     print("I_array:", p_arr.shape, np.sum(p_arr))
#     h = 0
#     for pi in p_arr:
#         h += entropy(pi / p_sum)
#     hr = h / np.log2(p_arr.shape[0])
#     print("H:", h)
#     print("H / log(q):", hr)
#     return hr
#
#
# def distance(patterns, pattern):
#     dis = Opt.distance(patterns, pattern)
#     return dis
#
#
# def min_distance(patterns, pattern):
#     dis = Opt.distance(patterns, pattern)
#     return np.min(dis)
#
#
# def most_similar_ps(patterns, pattern):
#     _ps = np.array(patterns)
#     _p = np.array(pattern)
#     _distance = distance(_ps, _p)
#     _index = np.argsort(_distance)
#     pattern_list = []
#     for i in _index:
#         if _distance[i] == _distance[_index[0]]:
#             pattern_list.append(patterns[i])
#     return pattern_list
#
#
# def nz_relu_sphere(patterns):
#     _one = np.ones([len(patterns), 1])
#     _ps = np.concatenate([_one, np.array(patterns)], axis=1)
#     _ps = np.maximum(_ps, 0)
#     _norm = 1 / np.sqrt(np.sum(np.square(_ps), axis=1))
#     _output = np.multiply(_ps, np.array([_norm]).T)
#     return _output.tolist()
#
#
# def sign(patterns):
#     return np.sign(patterns)
#
#
# def entropy(p):
#     if p == 0 or p == 1:
#         return 0
#     else:
#         ent = -1 * (p * np.log2(p) + (1-p) * np.log2(1-p))
#         return ent


class Cell:
    def __init__(self, pd):
        self.k = np.random.normal(0, 0.001, [pd, 1]).tolist()
        self.b = np.random.normal(0, 0.001)
        self.scale = 1

    def set_kb(self, k, b):
        norm = self.scale * np.sqrt(len(k)) / np.sqrt(np.sum(np.square(k)))
        # norm = 1 / np.sqrt(np.sum(np.square(k)))      # 严重影响精度？（已验证，影响sign精度）
        self.k = (k * norm).tolist()
        self.b = float(b * norm)
        # self.k = k.tolist()
        # self.b = float(b)

    def norm(self):
        norm = self.scale * np.sqrt(len(self.k)) / np.sqrt(np.sum(np.square(self.k)))
        # norm = 1 / np.sqrt(np.sum(np.square(self.k)))
        self.k = (np.array(self.k) * norm).tolist()
        self.b = float(self.b * norm)


class Layer:
    def __init__(self, layer_number, pattern_dimension, cell_number=0):
        self.ln = layer_number
        self.pd = pattern_dimension
        self.cells = []
        if cell_number != 0:
            for ci in range(cell_number):
                self.add_cell()

    def add_cell(self):
        self.cells.append(Cell(self.pd))

    # def batch_forwarding(self, patterns, is_last_layer):
    #     k, b = self.get_kb()
    #     if is_last_layer:
    #         _output = np.add(np.matmul(patterns, k), b)
    #     else:
    #         _output = sign(np.add(np.matmul(patterns, k), b))
    #     return _output

    def get_kb(self):
        k = np.array(self.cells[0].k)
        b = np.array(self.cells[0].b)
        if k is None or b is None:
            return None, None
        for i in range(len(self.cells)):
            if i > 0:
                ki = np.array(self.cells[i].k)
                bi = np.array(self.cells[i].b)
                if ki is None or bi is None:
                    return None, None
                k = np.append(k, ki, axis=1)
                b = np.append(b, bi)
        return k.tolist(), b.tolist()

    def set_kb(self, k, b):
        k = np.array(k)
        for i in range(len(self.cells)):
            ki = k[:, i:i+1]
            self.cells[i].set_kb(ki, b[i])
            # self.cells[i].k = ki.tolist()
            # self.cells[i].b = float(b[i])


class Network:
    def __init__(self, input_dimension, output_dimension, save_path):
        self.save_path = save_path
        self.id = input_dimension
        self.od = output_dimension
        self.layers = [Layer(0, pattern_dimension=self.id, cell_number=self.od)]
        self.layercount = 1
        self.architecture = [input_dimension, output_dimension]
        # self.mask = [15, 5]
        self.mask = None
        self.loop = 1000
        self.debug_trained_number = 0
        self.debug_add_cell_time = []
        self.debug_test_acc = []
        self.in_memory_data = []
        self.out_memory_data = []
        self.in_memory_labels = []
        self.out_memory_labels = []
        self.shift = 0
        self.debug_origin_k = []
        self.debug_origin_b = []

    def add_layer(self):
        print("Add layer")
        self.layers.append(Layer(self.layercount, self.od, self.od))
        self.layercount += 1
        self.architecture.append(self.od)
        if self.mask is not None:
            self.mask.append(self.mask[-1])
        new_k = np.zeros([self.od, self.od])
        new_b = np.zeros(self.od)
        for i in range(self.od):
            new_k[i, i] = 1
        self.layers[-1].set_kb(new_k, new_b)

    def add_cell(self, ln):
        print("Add Cell at layer", ln)
        self.layers[ln].add_cell()
        if ln < len(self.architecture) - 2:
            self.layers[ln+1].pd += 1
        self.architecture[ln+1] += 1
        self.debug_add_cell_time.append([len(self.in_memory_labels), ln])

    def one_pass(self, ln):
        print("One-pass: Add Cell at layer", ln)
        _k, _b = self.layers[ln].get_kb()
        self.layers[ln].add_cell()
        self.layers[ln + 1].pd += 1
        self.architecture[ln + 1] += 1
        self.debug_add_cell_time.append([len(self.in_memory_labels), ln])
        w = np.array(_k)
        w = np.concatenate([w, np.zeros([1, w.shape[1]])], axis=0)
        new = np.zeros([w.shape[0], 1])
        new[-1, 0] = 1
        new_w = np.concatenate([w, new], axis=1).tolist()
        _b.append(0)
        self.layers[ln].set_kb(new_w, _b)

    def get_ws_bs(self):
        ws = []
        bs = []
        for li in range(self.layercount):
            _k, _b = self.layers[li].get_kb()
            ws.append(_k)
            bs.append(_b)
        return ws, bs

    def set_ws_bs(self, ws, bs):
        for li in range(self.layercount):
            self.layers[li].set_kb(ws[li], bs[li])

    def get_path(self, data):
        ws, bs = self.get_ws_bs()
        return Tn.forwarding(self.architecture, ws, bs, data)

    # def batch_forwarding(self, patterns):
    #     path = [nz_relu_sphere(patterns)]
    #     for layer in self.layers:
    #         if layer.ln == self.layercount - 1:
    #             # ps = layer.batch_forwarding(path[layer.ln], True)
    #             ps = np.array(layer.batch_forwarding(path[layer.ln], False))[:, 1:].tolist()
    #         else:
    #             ps = layer.batch_forwarding(path[layer.ln], False)
    #         path.append(ps)
    #     hot = np.argmax(path[-1], axis=1)
    #     one_hot = np.zeros([len(patterns), self.architecture[-1]])
    #     for hi in range(len(hot)):
    #         one_hot[hi, hot[hi]] = 1
    #     path[-1] = one_hot
    #     return path

    # def adjust_layer(self):
    #     print("adjust_layer")
    #     half_size = 100
    #     loop = 1
    #     while loop <= self.loop:
    #         _data = []
    #         _labels = []
    #         for i in range(half_size):
    #             r = np.random.randint(0, len(self.out_memory_labels))
    #             _data.append(self.out_memory_data[r])
    #             _labels.append(self.out_memory_labels[r])
    #             del self.out_memory_data[r]
    #             del self.out_memory_labels[r]
    #         for i in range(half_size):
    #             r = np.random.randint(0, len(self.in_memory_labels))
    #             _data.append(self.in_memory_data[r])
    #             _labels.append(self.in_memory_labels[r])
    #             del self.in_memory_data[r]
    #             del self.in_memory_labels[r]
    #         ws, bs = self.get_ws_bs()
    #         new_ws, new_bs = Tn.train(self.architecture, ws, bs, _data, _labels, loop)
    #         append_list, del_list = self.test(_data, _labels, new_ws, new_bs)
    #         test_append_list, test_del_list = self.test(self.in_memory_data, self.in_memory_labels, new_ws, new_bs)
    #         print("correct:", len(append_list), "loss:", len(test_append_list) - len(self.in_memory_data))
    #         # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #         if len(append_list) >= len(test_del_list) + half_size:
    #             self.set_ws_bs(new_ws, new_bs)
    #             for oai in test_del_list:
    #                 self.out_memory_data.append(self.in_memory_data[oai])
    #                 self.out_memory_labels.append(self.in_memory_labels[oai])
    #             for di in range(len(test_del_list)):
    #                 del self.in_memory_data[test_del_list[-1 - di]]
    #                 del self.in_memory_labels[test_del_list[-1 - di]]
    #             for ai in append_list:
    #                 self.in_memory_data.append(_data[ai])
    #                 self.in_memory_labels.append(_labels[ai])
    #             for odi in del_list:
    #                 self.out_memory_data.append(_data[odi])
    #                 self.out_memory_labels.append(_labels[odi])
    #             loop += 1
    #         else:
    #             self.in_memory_data = self.in_memory_data + _data[half_size:]
    #             self.in_memory_labels = self.in_memory_labels + _labels[half_size:]
    #             self.out_memory_data = self.out_memory_data + _data[:half_size]
    #             self.out_memory_labels = self.out_memory_labels + _labels[:half_size]
    #             loop = np.maximum(1, loop - 1)
    #         print("loop:", loop, "total trained:", len(self.in_memory_labels), "+", len(self.out_memory_labels))

    def fine_tune(self, data, labels, loop, batch_size, lr=1e-6):
        print("fine_tuning")
        ws, bs = self.get_ws_bs()
        new_ws, new_bs = Tn.fine_tune(self.architecture, ws, bs, data, labels, batch_size, loop, lr, mask=self.mask)
        self.set_ws_bs(new_ws, new_bs)

    # def try_fine_tune(self, loop, batch_size, ws=None, bs=None, lr=1e-4, threshold=None):
    #     print("try_fine_tune")
    #     data = self.in_memory_data + self.out_memory_data
    #     labels = self.in_memory_labels + self.out_memory_labels
    #     if ws is None or bs is None:
    #         ws, bs = self.get_ws_bs()
    #     if threshold is None:
    #         a_list, d_list = self.test(data, labels)
    #         threshold = len(a_list)
    #     # new_ws, new_bs = Tn.fine_tune(self.architecture, ws, bs, data, labels, batch_size, loop, lr)
    #     new_ws, new_bs = Tn.fine_tune(self.architecture, ws, bs, self.in_memory_data, self.in_memory_labels, batch_size, loop, lr)
    #     a_list2, d_list2 = self.test(data, labels, new_ws, new_bs)
    #     # if add_data and add_labels:
    #     #     a_list3, d_list3 = self.test(add_data, add_labels, new_ws, new_bs)
    #     #     add_len = len(a_list3)
    #     # else:
    #     #     add_len = 0
    #     # print("add_len", add_len)
    #     if len(a_list2) >= threshold:
    #         self.set_ws_bs(new_ws, new_bs)
    #         self.in_memory_data = np.array(data)[a_list2].tolist()
    #         self.in_memory_labels = np.array(labels)[a_list2].tolist()
    #         self.out_memory_data = np.array(data)[d_list2].tolist()
    #         self.out_memory_labels = np.array(labels)[d_list2].tolist()
    #         print("try_fine_tune_result", len(a_list2), len(self.in_memory_data), len(d_list2), len(self.out_memory_data))
    #         return True, new_ws, new_bs
    #     else:
    #         print("try_fine_tune_fail")
    #         return False, new_ws, new_bs

    def check_memory(self):
        if self.mask is None:
            label_shift = None
        else:
            label_shift = [self.mask[-1], self.architecture[-1]]
        test_append_list, test_del_list = self.test(self.in_memory_data, self.in_memory_labels, label_shift=label_shift)
        for oai in test_del_list:
            self.out_memory_data.append(self.in_memory_data[oai])
            self.out_memory_labels.append(self.in_memory_labels[oai])
        for di in range(len(test_del_list)):
            del self.in_memory_data[test_del_list[-1 - di]]
            del self.in_memory_labels[test_del_list[-1 - di]]
        print("checked_memory, del", len(test_del_list))

    def unique_memory(self):
        # checking the same patterns
        print("len of in memory(before):", len(self.in_memory_data), len(self.in_memory_labels))
        temp_in_data, unique_ind = np.unique(self.in_memory_data, axis=0, return_index=True)
        temp_in_labels = []
        for i in unique_ind:
            temp_in_labels.append(self.in_memory_labels[i])
        self.in_memory_data = temp_in_data.tolist()
        self.in_memory_labels = temp_in_labels
        print("len of in memory(after):", len(self.in_memory_data), len(self.in_memory_labels))

    def test(self, data, labels, ws=None, bs=None, save=False, softmax=False, label_shift=None):
        if ws is None or bs is None:
            ws, bs = self.get_ws_bs()
        if len(data) == 0:
            return [], []
        else:
            test_append_list, test_del_list = Tn.test(self.architecture, ws, bs, data, labels,
                                                      softmax=softmax, label_shift=label_shift)
            if save:
                self.debug_test_acc.append([len(self.in_memory_labels), len(test_append_list) / len(data)])
            return test_append_list, test_del_list

    def forwarding(self, data, soft=False):
        ws, bs = self.get_ws_bs()
        pred = Tn.forwarding(self.architecture, ws, bs, data, soft=soft)
        return pred

    # def most_diff(self, path):
    #     _path = copy.deepcopy(path)
    #     _path[0] = np.maximum(np.sign(_path[0]), 0)
    #     diff = []
    #     for i in range(len(_path)):
    #         path_mean = np.mean(_path[i], axis=0)
    #         temp = np.array(_path[i]) - path_mean
    #         print(temp.shape)
    #         diff.append(np.sum(np.square(np.array(_path[i]) - path_mean)) / (len(_path[i]) * len(_path[i][0])))
    #     print(diff)
    #     for i in range(len(_path) - 1):
    #         if diff[i] < diff[i+1]:
    #             return i
    #     return len(_path) - 1
    #
    # def most_entropy(self, path):
    #     _path = copy.deepcopy(path)
    #     _path[0] = np.maximum(np.sign(_path[0]), 0)
    #     ent_list = []
    #     for i in range(len(_path)):
    #         p_list = np.sum(_path[i], axis=0) / len(_path[i])
    #         ent_sum = 0
    #         length = len(p_list)
    #         for j in range(len(p_list)):
    #             ent_v = entropy(p_list[j])
    #             if ent_v == 0:
    #                 length -= 1
    #             ent_sum += ent_v
    #         ent_list.append(ent_sum)
    #     print("entropy:", ent_list)
    #     for i in range(len(_path) - 1):
    #         if ent_list[i] > ent_list[i + 1]:
    #             return i
    #     return -1

    def adj_arc(self):
        if self.mask is None:
            label_shift = None
        else:
            label_shift = [self.mask[-1], self.architecture[-1]]

        patterns_d = self.out_memory_data
        labels_d = self.out_memory_labels
        self.out_memory_data = []
        self.out_memory_labels = []
        ###########################################################################################################
        # random_list = np.random.randint(0, 100, int(len(self.in_memory_data) // 100))
        # for i in range(len(random_list)):
        #     random_list[i] += 100 * i
        # arr_r = np.array(self.in_memory_data)[random_list]
        # path = Tn.forwarding(self.architecture, ws, bs, arr_r)
        # idh_list = []
        # for i in range(len(path)-1):
        #     ei = mutual_information(path[i], path[i+1])
        #     idh_list.append(ei)
        # print("mi / H random:", idh_list)
        ###########################################################################################################
        r_int = np.random.randint(0, len(labels_d))
        random_class = np.argmax(labels_d[r_int])
        print(np.sum(labels_d, axis=0), random_class)
        patterns_r = []
        labels_r = []
        for ri in range(len(patterns_d)):
            if np.argmax(labels_d[ri]) == random_class:
                patterns_r.append(patterns_d[ri])
                labels_r.append(labels_d[ri])
            else:
                self.out_memory_data.append(patterns_d[ri])
                self.out_memory_labels.append(labels_d[ri])
        ################################################################
        # 选层
        ws, bs = self.get_ws_bs()
        data_exp_rand_class = []
        for li in range(len(self.in_memory_labels)):
            if np.argmax(self.in_memory_labels[li]) != random_class:
                data_exp_rand_class.append(self.in_memory_data[li])
        in_path = Tn.forwarding(self.architecture, ws, bs, data_exp_rand_class)
        path = Tn.forwarding(self.architecture, ws, bs, patterns_r)

        bottleneck_layer, new_list, new_k, new_b = Tn.max_divide(in_path, path)
        # new_list, new_k, new_b = Tn.layer_max_divide(in_path, path, bottleneck_layer)
        if new_k is None:
            self.out_memory_data = self.out_memory_data + patterns_r
            self.out_memory_labels = self.out_memory_labels + labels_r
            self.loop *= 2
            self.try_fine_tuning(loop=40000, alldata=True)
            return
        adj_data = np.array(patterns_r)[new_list].tolist()
        adj_labels = np.array(labels_r)[new_list].tolist()

        # 优化方法得不到新cell的权值，转为数学方法直接解出
        # if loss_list[bottleneck_layer] * (-1) <= 0:
        #     sphere_in_path = nz_relu_sphere(in_path[bottleneck_layer])
        #     sphere_path = nz_relu_sphere(path[bottleneck_layer])
        #     max_dis = np.max(np.matmul(sphere_in_path, np.array(sphere_path).T), 0)
        #     min_pos = np.argmin(max_dis)
        #     new_k = np.array([sphere_path[min_pos]]).T      # / np.sum(np.square(path[bottleneck_layer]))
        #     new_b = (np.matmul([sphere_path[min_pos]], new_k)[0, 0]
        #              + np.max(np.matmul(sphere_in_path, new_k))) / (-2)
        #     adj_data, adj_labels = [patterns_d[min_pos]], [labels_d[min_pos]]
        #     threshold = 0
        # else:
        #     new_k = np.array(k_list[bottleneck_layer])
        #     new_b = b_list[bottleneck_layer]
        #     adj_data = np.array(self.in_memory_data)[new_del_list[bottleneck_layer]]
        #     adj_labels = np.array(self.in_memory_labels)[new_del_list[bottleneck_layer]]
        #     adj_data = np.concatenate([adj_data, np.array(patterns_d)[new_append_list[bottleneck_layer]]], axis=0).tolist()
        #     adj_labels = np.concatenate([adj_labels, np.array(labels_d)[new_append_list[bottleneck_layer]]], axis=0).tolist()
        #     threshold = len(new_del_list[bottleneck_layer])
        #     for di in range(threshold):
        #         patterns_d.append(self.in_memory_data[new_del_list[bottleneck_layer][-1 - di]])
        #         labels_d.append(self.in_memory_labels[new_del_list[bottleneck_layer][-1 - di]])
        #         del self.in_memory_data[new_del_list[bottleneck_layer][-1 - di]]
        #         del self.in_memory_labels[new_del_list[bottleneck_layer][-1 - di]]
        #     print("checker del new del list:", len(new_del_list[bottleneck_layer]), len(self.in_memory_data))
        ######################################################################################################
        # at last layer
        if bottleneck_layer == len(self.architecture) - 2:
            self.add_layer()
            # print("self.add_layer() beark")
            # exit()
            # self.fine_tune(self.in_memory_data, self.in_memory_labels, 60000, 1000, lr=1e-3)
            # return
            # try fine tune ###########################################################################
            # self.out_memory_data = self.out_memory_data + patterns_r
            # self.out_memory_labels = self.out_memory_labels + labels_r
            # sw = False
            # ws_try, bs_try = self.get_ws_bs()
            # data = self.in_memory_data + self.out_memory_data
            # labels = self.in_memory_labels + self.out_memory_labels
            # a_all, d_all = self.test(data, labels)
            # threshold = len(a_all)
            # while sw is False:
            #     sw, ws_try, bs_try = self.try_fine_tune(1000, 6000, ws_try, bs_try, 1e-3, threshold)
            # self.set_ws_bs(ws_try, bs_try)
            # patterns_r = self.out_memory_data
            # labels_r = self.out_memory_labels
            # self.out_memory_data = []
            # self.out_memory_labels = []
            # try fine tuen end #######################################################################

        # ws, bs = self.get_ws_bs()
        self.add_cell(bottleneck_layer)
        # self.layers[bottleneck_layer].cells[-1].k = new_k.tolist()
        # self.layers[bottleneck_layer].cells[-1].b = new_b
        self.layers[bottleneck_layer].cells[-1].set_kb(new_k, new_b)
        # kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk
        self.debug_origin_k.append((np.array(new_k).tolist(), bottleneck_layer, self.architecture[bottleneck_layer + 1] - 1))
        self.debug_origin_b.append((float(new_b), bottleneck_layer, self.architecture[bottleneck_layer + 1] - 1))

        ##########################################################################################################
        # target = Tn.find_target(self.architecture, ws, bs, self.in_memory_data, adj_labels[0], next_layer,
        #                         label_shift=label_shift)
        target = None
        next_layer = bottleneck_layer + 1
        while target is None:
            # 防止next_layer溢出，当下一层超出最后一层，target为标签
            # if next_layer >= len(path) - 2:
            #     target = adj_labels[0]
            # else:
            ws, bs = self.get_ws_bs()
            if self.mask is not None:
                label_shift = [self.mask[-1], self.architecture[-1]]
                target = Tn.target_prop(self.architecture, ws, bs, path[next_layer+1],
                                        adj_labels[0], next_layer, label_shift=label_shift)
            else:
                target = Tn.target_prop(self.architecture, ws, bs, path[next_layer + 1], adj_labels[0], next_layer)
            if target is None:
                self.one_pass(next_layer)
                next_layer += 1
        #########################################################################################################
        # DO NOT DEL THIS LINE: ws,bs = ...
        ws, bs = self.get_ws_bs()
        # ws_new, bs_new = Tn.train_next_layer(self.architecture, ws, bs, next_layer - 1,
        #                                      adj_data, target, mask=self.mask)
        ws_new, bs_new = Tn.train_new_parameters(self.architecture, ws, bs, next_layer - 1,
                                                 adj_data, adj_labels, mask=self.mask)

        # exit()
        # 训练无果，转为数学解法
        while ws_new is None or bs_new is None:
            # sphere_in_path = nz_relu_sphere(in_path[bottleneck_layer])
            # sphere_path = nz_relu_sphere(path[bottleneck_layer])
            # max_dis = np.max(np.matmul(sphere_in_path, np.array(sphere_path).T), 0)
            # min_pos = np.argmin(max_dis)
            # new_k = np.array([sphere_path[min_pos]]).T  # / np.sum(np.square(path[bottleneck_layer]))
            # new_b = (np.matmul([sphere_path[min_pos]], new_k)[0, 0]
            #          + np.max(np.matmul(sphere_in_path, new_k))) / (-2)
            # self.layers[bottleneck_layer].cells[-1].k = new_k.tolist()
            # self.layers[bottleneck_layer].cells[-1].b = new_b
            # ws_new, bs_new = Tn.train_new_cell(self.architecture, ws, bs, bottleneck_layer,
            #                                    [patterns_d[min_pos]], [labels_d[min_pos]], 0)
            exit()
        self.set_ws_bs(ws_new, bs_new)
        append_list2, del_list2 = self.test(patterns_r, labels_r, ws_new, bs_new, label_shift=label_shift)
        last_len = len(self.in_memory_labels)
        for ai in append_list2:
            self.in_memory_data.append(patterns_r[ai])
            self.in_memory_labels.append(labels_r[ai])
        for di in del_list2:
            self.out_memory_data.append(patterns_r[di])
            self.out_memory_labels.append(labels_r[di])
        print("new arc content:", len(append_list2))
        self.check_memory()
        if self.shift > len(self.in_memory_labels) - last_len:
            self.loop += 1000
        self.shift = len(self.in_memory_labels) - last_len
        print("shift:", self.shift)
        print("new architecture:", self.architecture)
        # self.try_fine_tune(1000, 1000)
        # self.save()
        print("total trained:", len(self.in_memory_labels), "+", len(self.out_memory_labels))
        return

    def training_sample(self, data_id, shift=False):
        temp_memory_data = self.out_memory_data
        temp_memory_labels = self.out_memory_labels
        self.out_memory_data = [temp_memory_data[0]]
        self.out_memory_labels = [temp_memory_labels[0]]
        self.debug_trained_number = data_id
        patterns = self.out_memory_data
        labels = self.out_memory_labels
        self.out_memory_data = []
        self.out_memory_labels = []
        new_ws, new_bs = self.get_ws_bs()
        batchsize = len(patterns) + len(self.in_memory_data)
        print("batchsize", batchsize)
        # for i in range(10):
        #     new_ws, new_bs = Tn.fine_tune(self.architecture, new_ws, new_bs, self.in_memory_data + patterns, self.in_memory_labels + labels,
        #                                   batchsize, 1000, 1e-3, mask=self.mask)
        #     if self.mask is None:
        #         label_shift = None
        #     else:
        #         label_shift = [self.mask[-1], self.architecture[-1]]
        #     append_list, del_list = self.test(patterns, labels, new_ws, new_bs, label_shift=label_shift)
        #     test_append_list, test_del_list = self.test(self.in_memory_data, self.in_memory_labels,
        #                                                 new_ws, new_bs, label_shift=label_shift)
        #     if len(del_list) + len(test_del_list) == 0:
        #         break
        for i in range(15):
            new_ws, new_bs = Tn.fine_tune(self.architecture, new_ws, new_bs, self.in_memory_data + patterns, self.in_memory_labels + labels,
                                          batchsize, 1000, 1e-4, mask=self.mask)
            if self.mask is None:
                label_shift = None
            else:
                label_shift = [self.mask[-1], self.architecture[-1]]
            append_list, del_list = self.test(patterns, labels, new_ws, new_bs, label_shift=label_shift)
            test_append_list, test_del_list = self.test(self.in_memory_data, self.in_memory_labels,
                                                        new_ws, new_bs, label_shift=label_shift)
            if len(del_list) + len(test_del_list) == 0:
                break
        print("train set correct:", len(append_list), "memory correct:", len(test_append_list), " / ",
              len(self.in_memory_data))
        if shift:
            shift = self.shift
        else:
            shift = 0
        if len(append_list) > len(test_del_list) + shift:
            self.shift = len(append_list) - len(test_del_list)
            self.loop = int(np.maximum(self.loop / 2, 1000))
            self.set_ws_bs(new_ws, new_bs)
            for oai in test_del_list:
                self.out_memory_data.append(self.in_memory_data[oai])
                self.out_memory_labels.append(self.in_memory_labels[oai])
            for di in range(len(test_del_list)):
                del self.in_memory_data[test_del_list[-1 - di]]
                del self.in_memory_labels[test_del_list[-1 - di]]
            for ai in append_list:
                self.in_memory_data.append(patterns[ai])
                self.in_memory_labels.append(labels[ai])
            for odi in del_list:
                self.out_memory_data.append(patterns[odi])
                self.out_memory_labels.append(labels[odi])

            print("total trained:", len(self.in_memory_labels), "+", len(self.out_memory_labels))
            self.out_memory_data = temp_memory_data[1:]
            self.out_memory_labels = temp_memory_labels[1:]
            return
        else:
            if len(patterns) < len(test_del_list):
                self.loop = int(np.maximum(self.loop / 2, 1000))
            self.adj_arc(patterns, labels)
            self.out_memory_data = temp_memory_data[1:]
            self.out_memory_labels = temp_memory_labels[1:]
            return

    def try_fine_tuning(self, loop=1000, alldata=False, shift=False):
        print("Try_fine_tuning...")
        patterns = self.out_memory_data
        labels = self.out_memory_labels
        self.out_memory_data = []
        self.out_memory_labels = []
        ws, bs = self.get_ws_bs()
        batchsize = int(np.ceil(len(self.in_memory_data) / 50))
        if alldata:
            _data = self.in_memory_data + patterns
            _labels = self.in_memory_labels + labels

            new_ws, new_bs = Tn.fine_tune(self.architecture, ws, bs, _data, _labels,
                                          batchsize, loop, 1e-3, mask=self.mask)
        else:
            new_ws, new_bs = Tn.fine_tune(self.architecture, ws, bs, self.in_memory_data, self.in_memory_labels,
                                          batchsize, loop, 1e-3, mask=self.mask)
        ###########################################################################################################
        # new_ws, new_bs = self.get_ws_bs()
        ##########################################################################################################
        if self.mask is None:
            label_shift = None
        else:
            label_shift = [self.mask[-1], self.architecture[-1]]
        append_list, del_list = self.test(patterns, labels, new_ws, new_bs, label_shift=label_shift)
        test_append_list, test_del_list = self.test(self.in_memory_data, self.in_memory_labels,
                                                    new_ws, new_bs, label_shift=label_shift)
        print("train set correct:", len(append_list), "memory correct:", len(test_append_list), " / ",
              len(self.in_memory_data))
        if shift:
            shift = self.shift
        else:
            shift = 0
        if len(append_list) > len(test_del_list) + shift:
            print("accept new parameters")
            self.shift = len(append_list) - len(test_del_list)
            self.loop = int(np.maximum(self.loop / 2, 1000))
            self.set_ws_bs(new_ws, new_bs)
            for oai in test_del_list:
                self.out_memory_data.append(self.in_memory_data[oai])
                self.out_memory_labels.append(self.in_memory_labels[oai])
            for di in range(len(test_del_list)):
                del self.in_memory_data[test_del_list[-1 - di]]
                del self.in_memory_labels[test_del_list[-1 - di]]
            for ai in append_list:
                self.in_memory_data.append(patterns[ai])
                self.in_memory_labels.append(labels[ai])
            for odi in del_list:
                self.out_memory_data.append(patterns[odi])
                self.out_memory_labels.append(labels[odi])

            print("total trained:", len(self.in_memory_labels), "+", len(self.out_memory_labels))
            return True
        else:
            print("reject new parameters")
            if len(patterns) < len(test_del_list):
                self.loop = int(np.maximum(self.loop / 2, 1000))
            self.out_memory_data = patterns
            self.out_memory_labels = labels
            return False

    def training(self, data_id=0):
        self.debug_trained_number = data_id
        acpt = self.try_fine_tuning()
        if acpt:
            return
        else:
            self.adj_arc()

    def save(self, ver_str=None):
        print("Saving!")
        if ver_str:
            file = open(self.save_path + ver_str, 'w', encoding='utf-8')
        else:
            file = open(self.save_path, 'w', encoding='utf-8')
        json_layers = []
        for li in self.layers:
            json_cells = []
            for ci in li.cells:
                if ci.b is not None:
                    tempb = ci.b
                else:
                    tempb = None
                if ci.k is not None:
                    tempk = ci.k
                else:
                    tempk = None
                json_cell = {"k": tempk,
                             "b": tempb,
                             "scale": ci.scale
                             }
                json_cells += [json_cell]
            json_layer = {"cells": json_cells,
                          "ln": li.ln,
                          "pd": li.pd
                          }
            json_layers += [json_layer]
        json_net = {"architecture": self.architecture, "layers": json_layers, "loop": self.loop,
                    "debug_trained_number": self.debug_trained_number, "debug_add_cell_time": self.debug_add_cell_time,
                    "in_memory_data": self.in_memory_data, "in_memory_labels": self.in_memory_labels,
                    "out_memory_data": self.out_memory_data, "out_memory_labels": self.out_memory_labels,
                    "debug_test_acc": self.debug_test_acc, "shift": self.shift, "mask": self.mask,
                    "debug_origin_k": self.debug_origin_k, "debug_origin_b": self.debug_origin_b}
        json.dump(json_net, file)
        file.close()
        print("Net Saved.")

    def load(self):
        print("Loading...")
        file = open(self.save_path, 'r', encoding='utf-8')
        data = json.load(file)
        architecture = data["architecture"]
        while len(self.architecture) < len(architecture):
            self.add_layer()
        for li in range(len(architecture) - 1):
            while len(self.layers[li].cells) < architecture[li + 1]:
                self.add_cell(li)
        print("architecture", self.architecture, architecture)
        # self.shift = data["shift"]
        self.loop = data["loop"]
        self.mask = data["mask"]
        # self.debug_origin_k = data["debug_origin_k"]
        # self.debug_origin_b = data["debug_origin_b"]
        # self.debug_trained_number = data["debug_trained_number"]
        self.debug_add_cell_time = data["debug_add_cell_time"]
        self.debug_test_acc = data["debug_test_acc"]
        self.in_memory_data = data["in_memory_data"]
        self.in_memory_labels = data["in_memory_labels"]
        self.out_memory_data = data["out_memory_data"]
        self.out_memory_labels = data["out_memory_labels"]
        ln = 0
        for li in self.layers:
            layer_data = data["layers"][ln]
            cells_data = layer_data["cells"]
            li.ln = layer_data["ln"]
            li.pd = layer_data["pd"]
            cn = 0
            for ci in li.cells:
                cell_data = cells_data[cn]
                ci.k = cell_data["k"]
                ci.b = cell_data["b"]
                # ci.set_kb(cell_data["k"], cell_data["b"])
                ci.norm()
                # ci.scale = cell_data["scale"]
                cn += 1
            ln += 1
        file.close()
        print("Loading parameters completed.")

    def construct(self):
        file = open(self.save_path, 'r', encoding='utf-8')
        data = json.load(file)
        architecture = data["architecture"]
        while len(self.architecture) < len(architecture):
            self.add_layer()
        for li in range(len(architecture) - 1):
            while len(self.layers[li].cells) < architecture[li + 1]:
                self.add_cell(li)
        print("architecture", self.architecture, architecture)
        self.in_memory_data = data["in_memory_data"]
        self.in_memory_labels = data["in_memory_labels"]
        self.out_memory_data = data["out_memory_data"]
        self.out_memory_labels = data["out_memory_labels"]
        ln = 0
        for li in self.layers:
            for ci in li.cells:
                ci.k = np.random.normal(0, 0.001, [self.architecture[ln], 1]).tolist()
                ci.b = np.random.normal(0, 0.001)
            ln += 1
        file.close()

    def debug_cos(self):
        cos_list = []
        for k in range(len(self.debug_origin_k)):
            k_o = self.debug_origin_k[k][0]
            ln = self.debug_origin_k[k][1]
            cn = self.debug_origin_k[k][2]
            k_n = self.layers[ln].cells[cn].k
            for i in range(len(k_n) - len(k_o)):
                k_o.append([0])
            cosi = np.matmul(np.array(k_o).T, k_n) / (np.sqrt(np.sum(np.square(k_o))) * np.sqrt(np.sum(np.square(k_n))))
            cos_list.append(np.sum(cosi))
        print(cos_list)
