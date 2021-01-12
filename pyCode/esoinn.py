# Copyright (c) 2017 Gangchen Hua
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php

"""
E-SOINN in Python 3
Version 1.0
"""

from typing import overload
import numpy as np
from scipy.sparse import dok_matrix
from sklearn.base import BaseEstimator, ClusterMixin
from random import choice
import matplotlib.pyplot as plt
import threading

# inheriting
class ESoinn(BaseEstimator, ClusterMixin):
    # nodes initially don't have a label, set it to -1
    INITIAL_LABEL = -1

    def __init__(self, dim=2, max_edge_age=50, iteration_threshold=200, c1=0.001, c2=1.0):
        '''
        Set hyperparameters.
        '''
        self.dim = dim
        self.iteration_threshold = iteration_threshold
        self.c1 = c1
        self.c2 = c2
        self.max_edge_age = max_edge_age
        self.num_signal = 0
        self._reset_state() # reset inside state
        self.fig = plt.figure()

    def _reset_state(self):
        self.nodes = np.array([], dtype=np.float64)
        # winning times of a node, aka. local accumulated number of signals
        self.winning_times = []
        self.density = []
        self.N = []
        # if active
        self.won = []
        self.total_loop = 1
        self.s = [] # accumulated points of nodes
        # edges are stored in key-value pair
        self.adjacent_mat = dok_matrix((0, 0), dtype=np.float64) 
        self.node_labels = [] # subclass labels
        self.labels_ = []
        self.sigs = []

    # this needs to be modified.
    # def fit(self, X):
    #     """
    #     train data in batch manner
    #     :param X: array-like or ndarray
    #     """
    #     self._reset_state()
    #     # choose 50000 signals randomly
    #     for x in range(50000):
    #         self.input_signal(choice(X))
    #     # self.labels_ = self.__label_samples(X)
    #     self.__classify()
    #     plt.show()
    #     return self
    
    def fit(self, X):
        """
        train data in batch manner
        :param X: array-like or ndarray
        """
        self._reset_state()
        # total number of samples
        total_sample_num = X.shape[0]
        # train in an online form
        for i in range(total_sample_num):
            # input sample
            x = X[i, :]
            self.input_signal(x)
        # self.labels_ = self.__label_samples(X)
        self.__classify()
        # plt.show()
        return self

    def input_signal(self, signal: np.ndarray):
        """
        Input a new signal one by one, which means training in online manner.
        fit() calls __init__() before training, which means resetting the
        state. So the function does batch training.
        :param signal: A new input signal
        :return:
        """
        # Algorithm 3.4 (2)
        signal = self.__check_signal(signal)
        # keep track of number of signals input so far
        self.num_signal += 1
        # Store the input signal?
        self.sigs.append(signal)

        # Algorithm 3.4 (1)
        # if number of nodes is smaller than 2, set the input signal as a node.
        if len(self.nodes) < 2:
            self.__add_node(signal)
            return

        # Algorithm 3.4 (3)
        winner, dists = self.__find_nearest_nodes(2, signal)
        sim_thresholds = self.__calculate_similarity_thresholds(winner)
        # signal is a new pattern 
        if dists[0] > sim_thresholds[0] or dists[1] > sim_thresholds[1]:
            self.__add_node(signal)
        else:
            # Algorithm 3.4 (4)
            self.__increment_edge_ages(winner[0])
            # Algorithm 3.4 (5)
            need_add_edge, need_combine = self.__need_add_edge(winner)
            if need_add_edge:
                # print("add edge")
                # Algorithm 3.4 (5)(a)
                self.__add_edge(winner)
            else:
                # Algorithm 3.4 (5)(b)
                self.__remove_edge_from_adjacent_mat(winner)
            # Algorithm 3.4 (5)(a) need to combine subclasses
            if need_combine:
                self.__combine_subclass(winner)
            # Algorithm 3.4 (6) checked, maybe fixed problem N
            self.__update_density(winner[0])
            # Algorithm 3.4 (7) is embedded in func __update_density()
            # Algorithm 3.4 (8) (a)
            self.__update_winner(winner[0], signal)
            # Algorithm 3.4 (8) (b)
            self.__update_adjacent_nodes(winner[0], signal)

        # Algorithm 3.4 (9)
        self.__remove_old_edges()

        # Algorithm 3.4 (10)
        if self.num_signal % self.iteration_threshold == 0 and self.num_signal > 1:
            # print(self.won)
            # update self.N based on self.won
            for i in range(len(self.won)):
                if self.won[i]:
                    self.N[i] += 1
            for i in range(len(self.won)):
                self.won[i] = False
            print("Input signal amount:", self.num_signal, "nodes amount:", len(self.nodes))
            self.__separate_subclass()
            self.__delete_noise_nodes()
            self.total_loop += 1 # ?
            # algo 3.4(11)?
            self.__classify()
            # plot
            # threading.Thread(self.plot_NN())
            
            # clear signals
            self.sigs.clear()

    # checked
    def __combine_subclass(self, winner):
        '''
        This function combines the subclasses that winner and second winner 
        belong to.
        INPUTS:
            winner: list, contains indices of winner and second winner.
        '''
        # when enter this function, winner and second winner belong to different
        # subclasses by default. Thus if  obey this rule, raise error.
        if self.node_labels[winner[0]] == self.node_labels[winner[1]]:
            raise ValueError
        # combine two subclasses as on class, with the label of winner's label.
        class_id = self.node_labels[winner[0]]
        node_belong_to_class_1 = self.find_all_index(self.node_labels, self.node_labels[winner[1]])
        for i in node_belong_to_class_1:
            self.node_labels[i] = class_id

    # checked
    def __remove_old_edges(self):
        '''
        This function removes the edges with age greater than age_max.
        '''
        for i in list(self.adjacent_mat.keys()):
            # greater than max_edge_age + 1 because initially the edge is set 
            # to 1 instead of 0
            if self.adjacent_mat[i] > self.max_edge_age + 1:
                # print("Edge removed")
                self.adjacent_mat.pop((i[0], i[1]))

    # checked
    def __remove_edge_from_adjacent_mat(self, ids):
        '''
        This function remove the edge between two nodes.
        INPUTS:
            ids: list, ids[0] is the index of node A, ids[1] is the index of 
            node B.
        '''
        if (ids[0], ids[1]) in self.adjacent_mat and (ids[1], ids[0]) in self.adjacent_mat:
            self.adjacent_mat.pop((ids[0], ids[1]))
            self.adjacent_mat.pop((ids[1], ids[0]))

    # Algorithm 3.1
    def __separate_subclass(self):
        '''
        This function implements algo 3.1 in the original paper.
        '''
        # find all local apex
        density_dict = {}
        # this is not necessary because self.density is already a list.
        density = list(self.density)
        for i in range(len(self.density)):
            density_dict[i] = density[i]
        class_id = 0
        while len(density_dict) > 0:
            # apex index
            apex = max(density_dict, key=lambda x: density_dict[x])
            # print("len", len(density_dict))
            # print("apex", apex)
            # after calling __get_nodes_by_apex, ids contains all the indices
            # of nodes that have the same apex
            ids = []
            ids.append(apex)
            self.__get_nodes_by_apex(apex, ids, density_dict)
            # classify all other nodes with the same subclass label as their apex
            for i in set(ids):
                if i not in density_dict:
                    raise ValueError
                self.node_labels[i] = class_id
                density_dict.pop(i)
            class_id += 1

    def __get_nodes_by_apex(self, apex, ids, density_dict):
        '''
        This is a recursive function, depth first search all the nodes that have 
        the same apex.
        INPUTS:
            apex: int, index of the root node.
            ids: list, contains indices of the searched nodes that have the same
            apex.
            density_dict: dict, key-value pair = index-density pair.
         
        '''
        new_ids = []
        pals = self.adjacent_mat[apex]
        # pals.keys()[1]: indices of nodes that are adjacent to apex.
        for k in pals.keys():
            i = k[1]
            if self.density[i] <= self.density[apex] and i in density_dict and i not in ids:
                ids.append(i)
                new_ids.append(i)
        if len(new_ids) != 0:
            for i in new_ids:
                self.__get_nodes_by_apex(i, ids, density_dict)
        else:
            return

    # Algorithm 3.2, checked
    def __need_add_edge(self, winner):
        '''
        This function determines whether add an edge between winner and second
        winner, and whether combine two subclasses given winner and second winner.
        INPUTS:
            winner: list, winner[0] is the index of the winner, winner[1] is 
            the index of the second winner.
        RETURNS:
            1. need_add_edge: bool, whether need to add an edge between winner
               and second winner.
            2. need_combine: bool, whether need to combine subclasses.
        '''
        # winner or second winner is a new node (it is not yet determined to 
        # which subclass the node belongs).
        if self.node_labels[winner[0]] == self.INITIAL_LABEL or \
                self.node_labels[winner[1]] == self.INITIAL_LABEL:
            # connect these two nodes, do not combine subclasses
            return True, False
        # winner and second winner belong to the same subclasses
        elif self.node_labels[winner[0]] == self.node_labels[winner[1]]:
            # connect these two nodes, not combine subclasses
            return True, False
        # winner and second winner belong to different subclasses
        else:
            # mean density and apex density of the subclass which winner belongs to
            mean_density_0, max_density_0 = self.__mean_max_density(self.node_labels[winner[0]])
            # mean density and apex density of the subclass which second winner belongs to
            mean_density_1, max_density_1 = self.__mean_max_density(self.node_labels[winner[1]])
            alpha_0 = self.calculate_alpha(mean_density_0, max_density_0)
            alpha_1 = self.calculate_alpha(mean_density_1, max_density_1)
            # find the minimum density between winner and second winner
            min_density = min([self.density[winner[0]], self.density[winner[1]]])
            # print(self.density[winner[0]], self.density[winner[1]])
            # print(mean_density_0, max_density_0, mean_density_1, max_density_1, alpha_0, alpha_1, min_density)
            # Eq (7) and (8) in the original paper
            if alpha_0 * max_density_0 < min_density or alpha_1 * max_density_1 < min_density:
                # print("True")
                # connect these two nodes, combine subclasses
                return True, True
            else:
                # not connect these two nodes, not combine subclasses
                return False, False
  
    #一般来说，要使用某个类的方法，需要先实例化一个对象再调用方法。
    #而使用@staticmethod或@classmethod，就可以不需要实例化，直接类名.方法名()来调用
    @staticmethod
    def calculate_alpha(mean_density, max_density):
        '''
        This function computes the automatically determined parameter alpha.
        Equation (9) in the original paper.
        INPUTS:
            mean_density: float, mean density of a given subclass.
            max_density: float, apex density of a given subclass.
        RETURNS:
            alpha: float.
        '''
        if max_density > 3.0 * mean_density:
            return 1.0
        elif 2.0 * mean_density < max_density <= 3.0 * mean_density:
            return 0.5
        else:
            return 0.0

    @staticmethod
    def find_all_index(ob, item):
        '''
        This function finds all nodes index that belong to a specific class ID.
        INPUTS:
            ob: list, self.node_labels.
            item: int, the given class ID.
        RETURNS:
            a list of node indices that belongs to the given class ID.
        '''
        return [i for i, a in enumerate(ob) if a == item]

    # checked
    def __mean_max_density(self, class_id):
        '''
        This function computes mean density and apex density of a given class
        ID.
        INPUTS:
            class_id: int, the class ID of which we want to compute mean density
            and apex density.
        RETURNS:
            avg_density: float, mean density of subclass with class ID class_id.
            max_density: float, apex density of subclass with class ID class_id.
        '''
        node_belong_to_class = self.find_all_index(self.node_labels, class_id)
        avg_density = 0.0
        max_density = 0.0
        for i in node_belong_to_class:
            avg_density += self.density[i]
            if self.density[i] > max_density:
                max_density = self.density[i]
        avg_density /= len(node_belong_to_class)
        return avg_density, max_density

    @overload
    def __check_signal(self, signal: list) -> None:
        ...

    def __check_signal(self, signal: np.ndarray):
        """
        This function checkes type and dimensionality of an input signal.
        If signal is the very first input signal, set the dimension of it as
        self.dim. So, this method have to be called before calling functions
        that use self.dim.
        INPUTS:
            signal: expected to be numpy vec.
        RETURNS:
            signal: numpy arrray with shape(nFeature, )
        """
        # if signal is a list, convert it to nparray
        if isinstance(signal, list):
            signal = np.array(signal)
        # if signal is still not nparray, raise error
        if not (isinstance(signal, np.ndarray)):
            print("1") 
            # unknown type of input signal
            raise TypeError()
        # if signal is not a vector of shape(nFeature, )
        if len(signal.shape) != 1:
            print("2") 
            # input signal has to be a vector
            raise TypeError()
        # set self.dim
        self.dim = signal.shape[0]
        # if self still doesn't has the attribute 'dim', set it.
        if not (hasattr(self, 'dim')):
            self.dim = signal.shape[0]
        else:
            # if dim of signal doesn't match self.dim, raise error
            if signal.shape[0] != self.dim:
                print("3")
                raise TypeError()
        return signal

    # checked
    def __add_node(self, signal: np.ndarray):
        n = self.nodes.shape[0]
        self.nodes.resize((n + 1, self.dim))
        self.nodes[-1, :] = signal
        self.winning_times.append(1)
        self.adjacent_mat.resize((n + 1, n + 1))
        self.N.append(1) # ?
        self.density.append(0)
        self.s.append(0) # accumulated points
        self.won.append(False) # This is used for determining N
        # not classified to any exsisting subclasses
        self.node_labels.append(self.INITIAL_LABEL) 

    # checked
    def __find_nearest_nodes(self, num: int, signal: np.ndarray):
        '''
        This function finds the num nearest neighbor in the exsisting nodes
        of the input signal, and returns the indices of the num nearest nodes
        and num nearest distances.
        INPUTS:
            num: int, num nearest neighbors.
            signal: np array of shape (nFeature, ), input signal.
        RETURNS:
            indexes: list, contains the indices of num nearest neighbors.
            sq_dists: list, contains square distances of num nearest neighbors.
        '''
        n = self.nodes.shape[0]
        # find num nearest neighbors
        indexes = [0] * num
        sq_dists = [0.0] * num
        # maybe don't need to *n, Python can broadcast
        D = np.sum((self.nodes - np.array([signal] * n)) ** 2, 1)
        # print("D", D)
        # find num nearest neighbors
        for i in range(num):
            # except nan value, find the index
            indexes[i] = np.nanargmin(D)
            sq_dists[i] = D[indexes[i]]
            # remove this one
            D[indexes[i]] = float('nan')
        return indexes, sq_dists

    # checked
    def __calculate_similarity_thresholds(self, node_indexes):
        '''
        This function calculates the similarity thresholds of the nodes with 
        the given indices.
        INPUTS:
            node_indexes: list, contains indices of winning nodes.
        RETURNS:
            sim_thresholds: list, contains similarity thresholds of the given 
            winning nodes.
        '''
        sim_thresholds = []
        for i in node_indexes:
            pals = self.adjacent_mat[i, :]
            # if the elements of a sparse matrix are 0, 0's are not stored,
            # hence if node[i] doesn't have neighbors, len(pals) is 0.
            # node[i] is not connected with any nodes with an edge.
            if len(pals) == 0:
                # finding 2 nearest neighbors of node[i] (including itself),
                # the second neighbor is its nearest neighbor.
                # the minimum distance between node[i] and other nodes in the network
                idx, sq_dists = self.__find_nearest_nodes(2, self.nodes[i, :])
                sim_thresholds.append(sq_dists[1])
            
            # node[i] is connected to some other nodes.
            else:
                pal_indexes = []
                for k in pals.keys():
                    pal_indexes.append(k[1])
                sq_dists = np.sum((self.nodes[pal_indexes] - np.array([self.nodes[i]] * len(pal_indexes))) ** 2, 1)
                # the maximum distance between node[i] and its neighboring nodes
                sim_thresholds.append(np.max(sq_dists))
        return sim_thresholds

    # checked
    def __add_edge(self, node_indexes):
        '''
        This function adds an edge between node_indexes[0] and node_indexes[1].
        '''
        # Question: the original paper set the age of the new edge as 0!
        self.__set_edge_weight(node_indexes, 1)

    # checked
    def __increment_edge_ages(self, winner_index):
        for k, v in self.adjacent_mat[winner_index, :].items():
            self.__set_edge_weight((winner_index, k[1]), v + 1)

    # checked
    def __set_edge_weight(self, index, weight):
        # index[0] is the winner node, index[1] is the adjacent node
        # set twice because it is an undirected graph
        self.adjacent_mat[index[0], index[1]] = weight
        self.adjacent_mat[index[1], index[0]] = weight

    # checked
    def __update_winner(self, winner_index, signal):
        '''
        This function updates the weight of winner node according to algo 3.4(8)
        INPUTS:
            winner_index: int, index of the winner node.
            signal: np array of shape(nFeature, ), input signal.
        '''
        # retrieve weight vector of the winning point
        w = self.nodes[winner_index]
        self.nodes[winner_index] = w + (signal - w) / self.winning_times[winner_index]

    # checked, maybe fixed the N problem
    def __update_density(self, winner_index):
        '''
        This function updates the density of nodes according to Eq (6).
        INPUTS:
            winner_index: index of the first winner.
        '''
        # algo 3.4(7)
        self.winning_times[winner_index] += 1
        # if self.N[winner_index] == 0:
        #     raise ValueError
        # print(self.N[winner_index])
        pals = self.adjacent_mat[winner_index]
        pal_indexes = []
        # find the indices of the nodes that are adjacent to winner
        for k in pals.keys():
            pal_indexes.append(k[1])
        if len(pal_indexes) != 0:
            # print(len(pal_indexes))
            sq_dists = np.sum((self.nodes[pal_indexes] - np.array([self.nodes[winner_index]] * len(pal_indexes))) ** 2, 1)
            # print(sq_dists)
            # d_bar in original paper
            mean_adjacent_density = np.mean(np.sqrt(sq_dists))
            p = 1.0 / ((1.0 + mean_adjacent_density) ** 2)
            # accumulated points
            self.s[winner_index] += p
            if self.N[winner_index] == 0:
                self.density[winner_index] = self.s[winner_index]
            else:
                self.density[winner_index] = self.s[winner_index] / self.N[winner_index]
        # stepstone for calculating N
        if self.s[winner_index] > 0:
            self.won[winner_index] = True

    # checked
    def __update_adjacent_nodes(self, winner_index, signal):
        '''
        This function updates weights of nodes which are adjacent to winner
        according to algo 3.4(8).
        INPUTS:
            winner_index: int, index of the winner.
            signal: np array of shape(nFeature, ), input signal.
        '''
        pals = self.adjacent_mat[winner_index]
        for k in pals.keys():
            i = k[1]
            w = self.nodes[i]
            self.nodes[i] = w + (signal - w) / (100 * self.winning_times[i])

    # checked
    def __delete_nodes(self, indexes):
        # have no nodes to be deleted
        if not indexes:
            return
        n = len(self.winning_times)
        # this deletion changes the index of the remaining nodes
        self.nodes = np.delete(self.nodes, indexes, axis=0)
        # node indices that survived the deletion
        remained_indexes = list(set([i for i in range(n)]) - set(indexes))
        # the below deletion also changes the indices of remaining nodes
        self.winning_times = [self.winning_times[i] for i in remained_indexes]
        self.N = [self.N[i] for i in remained_indexes]
        self.density = [self.density[i] for i in remained_indexes]
        self.node_labels = [self.node_labels[i] for i in remained_indexes]
        self.won = [self.won[i] for i in remained_indexes]
        self.s = [self.s[i] for i in remained_indexes]
        # also have to delete edges from the adjacent matrix
        self.__delete_nodes_from_adjacent_mat(indexes, n, len(remained_indexes))

    # checked
    # cannot understand
    def __delete_nodes_from_adjacent_mat(self, indexes, prev_n, next_n):
        '''
        This function deletes edges from the adjacent matrix after deleting 
        the corresponding nodes.
        INPUTS:
            indexes: list, contains all the indices which corresponds to the 
            deleted nodes.
            prev_n: int, the number of nodes before deletion.
            next_n: int, the number of nodes after deletion.
        '''
        # looping until indexes is popped empty
        while indexes:
            
            next_adjacent_mat = dok_matrix((prev_n, prev_n))
            # (key1, key2) = (first node, second node) = an edge
            for key1, key2 in self.adjacent_mat.keys():
                if key1 == indexes[0] or key2 == indexes[0]:
                    # ignore the deleted node
                    continue
                if key1 > indexes[0]:
                    # adjust index for nodes behind the deleted node
                    new_key1 = key1 - 1
                else:
                    # hasn't reached the deleted node
                    new_key1 = key1
                if key2 > indexes[0]:
                    # adjust index for nodes behind the deleted node
                    new_key2 = key2 - 1
                else:
                    # hasn't reached the deleted node
                    new_key2 = key2
                # Because dok_matrix.__getitem__ is slow,
                # access as dictionary.
                next_adjacent_mat[new_key1, new_key2] = super(dok_matrix, self.adjacent_mat).__getitem__((key1, key2))
            self.adjacent_mat = next_adjacent_mat.copy()
            indexes = [i - 1 for i in indexes]
            indexes.pop(0)
        self.adjacent_mat.resize((next_n, next_n))

    # checked
    def __delete_noise_nodes(self):
        '''
        This function implements algo 3.4(10)(b), which deletes the noisy nodes.
        '''
        n = len(self.winning_times)
        # print(n)
        # indices of nodes that are about to be deleted
        noise_indexes = []
        mean_density_all = np.mean(self.density)
        # print(mean_density_all)
        for i in range(n):
            # case (1)
            if len(self.adjacent_mat[i, :]) == 2 and self.density[i] < self.c1 * mean_density_all:
                noise_indexes.append(i)
            # case (2)
            elif len(self.adjacent_mat[i, :]) == 1 and self.density[i] < self.c2 * mean_density_all:
                noise_indexes.append(i)
            # case (3)
            elif len(self.adjacent_mat[i, :]) == 0:
                noise_indexes.append(i)
        print("Removed noise node:", len(noise_indexes))
        self.__delete_nodes(noise_indexes)

    def __get_connected_node(self, index, indexes):
        '''
        This function finds all the nodes that are connected to the given node
        by a path.
        However, didn't reflect searching for the UNCLASSIFIED nodes.
        '''
        new_ids = []
        pals = self.adjacent_mat[index]
        for k in pals.keys():
            i = k[1]
            if i not in indexes:
                indexes.append(i)
                new_ids.append(i)

        if len(new_ids) != 0:
            for i in new_ids:
                self.__get_connected_node(i, indexes)
        else:
            return

    # Algorithm 3.3
    def __classify(self):
        '''
        This function implements algo 3.3 in the original paper.
        '''
        need_classified = list(range(len(self.node_labels)))
        # initialize all nodes as unclassified
        for i in range(len(self.node_labels)):
            self.node_labels[i] = self.INITIAL_LABEL
        class_id = 0
        while len(need_classified) > 0:
            indexes = []
            # randomly choose one unclassified node i from node set
            index = choice(need_classified)
            indexes.append(index)
            # search the node set  to find all unclassified nodes that are 
            # connected to node i with a path
            self.__get_connected_node(index, indexes)
            # mark these nodes as classified and label them as the same class
            # as node i
            for i in indexes:
                self.node_labels[i] = class_id
                need_classified.remove(i)
            class_id += 1

        print("Number of classes：", class_id)

    def plot_NN(self):
        plt.cla()
        # for k in self.sigs:
        #     plt.plot(k[0], k[1], 'cx')
        for k in self.adjacent_mat.keys():
            plt.plot(self.nodes[k, 0], self.nodes[k, 1], 'k', c='blue')
        # plt.plot(nodes[:, 0], nodes[:, 1], 'ro')

        color = ['black', 'red', 'saddlebrown', 'skyblue', 'magenta', 'green', 'gold']

        color_dict = {}

        for i in range(len(self.nodes)):
            if not self.node_labels[i] in color_dict:
                color_dict[self.node_labels[i]] = choice(color)
            plt.plot(self.nodes[i][0], self.nodes[i][1], 'ro', c=color_dict[self.node_labels[i]])

        plt.grid(True)
        plt.pause(0.05)
