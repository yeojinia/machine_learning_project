"""
Filename: 	    decision_tree.py
Author:   	    Yeojin Kim
Date:     	    November 2020
Description:    This file is an implementation of decision tree for binary classification.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import queue

class Data:
    """
    A class for data
    """
    def __init__(self, feature, classfied_y):
        self.X = pd.read_csv(feature)
        self.y = pd.read_csv(classfied_y, header=None)
        self.header_name = []
        for col in self.X.columns:
            self.header_name.append(col)

    def getX(self):
        return self.X

    def getY(self):
        return self.y

    def getHeaderLabel(self):
        return self.header_name

class Node:
    """
    A class for Tree Node
    """
    def __init__(self):
        self.depth = None
        self.feature = None
        self.left = None
        self.right = None
        self.label = None
        self.info_gain = None

class Tree:
    """
    A class for Tree
    """
    def __init__(self, data):
        self.data = data

    def get_entropy(self, count, total):
        """
        :param total:
        :return: entropy
        """
        if(count == 0):
            return 0
        return -(count*1.0/total)*math.log(count*1.0/total, 2)

    def information_gain(self, plus, minus, left_plus, left_minus, right_plus, right_minus):
        """
        :param plus: the number of nodes classified as +1
        :param minus: the number of nodes classified as -1
        :param left_plus: the number of left-child classified as +1
        :param left_minus: the number of left-child classified as -1
        :param right_plus: the number of right-child classified as +1
        :param right_minus: the number of right-child classified as -1
        :return: information gain measures the reduction of target variable
        """
        parent_uncertainty = self.get_entropy(plus, plus+ minus) + self.get_entropy(minus, plus+ minus)
        left_uncertainty = self.get_entropy(left_plus, left_plus+ left_minus) + self.get_entropy(left_minus, left_plus+ left_minus) # x[feature_idx] == 1
        right_uncertainty = self.get_entropy(right_plus, right_plus + right_minus) + self.get_entropy(right_minus, right_plus +right_minus) # x[feature_idx] == 0
        left_prob = (left_plus + left_minus)/(plus + minus)
        right_prob = (right_plus + right_minus) / (plus + minus)
        gain = left_prob*left_uncertainty + right_prob*right_uncertainty
        return (parent_uncertainty - gain)

    def split_criteria(self):
        """
        :return: best feature index, the information gain of the best feature index
        """
        plus = np.sum(self.data[self.data[:, -1]>=0][:,-1])
        minus = -np.sum(self.data[self.data[:, -1] <0][:,-1])

        information_gain = -1
        feature_idx = 0
        for idx in range(0, self.data.shape[1]-1):
            left_plus = 0
            left_minus = 0
            right_plus = 0
            right_minus = 0
            XYs = np.copy(self.data[:, [idx, -1]])
            for _xy in XYs:
                if(_xy[0] == 1 and _xy[1] > 0):
                    left_plus += _xy[1]
                elif(_xy[0] == 1 and _xy[1] <= 0):
                    left_minus -= _xy[1]
                elif(_xy[0] == 0 and _xy[1] > 0):
                    right_plus += _xy[1]
                elif(_xy[0] == 0 and _xy[1] <= 0):
                    right_minus -= _xy[1]

            if(plus + minus!=0):
                gain = self.information_gain(plus, minus, left_plus, left_minus, right_plus, right_minus)
                if(gain > information_gain):
                    feature_idx = idx
                    information_gain = gain

        return feature_idx, information_gain

class DecisionTree:
    """
    A class for Decision
    """
    def __init__(self, max_depth):
        """
        :param max_depth: maximum depth of decision tree
        """
        self.root = Node()
        self.max_depth = max_depth

    def fit(self, x, y, depth=0):
        """
        :param x: (pandas data frame) examples
        :param y: (pandas data frame) ground truth
        :param depth: start depth
        :return: root node
        """
        X = np.array(x)
        Y = np.array(y)
        data = np.hstack((X, Y))
        node = self.fit_node(data, 0)
        return node

    def fit_node(self, data, depth):
        """
        :param data: X(examples) + Y(ground truth)
        :param depth: current depth of the node
        :return: node
        """
        plus = np.sum(data[data[:, -1] >= 0][:, -1])
        minus = -np.sum(data[data[:, -1] < 0][:, -1])
        node = Node()
        node.depth = depth
        if(plus > minus):
            node.label = 1
        else:
            node.label = -1

        if(depth <= self.max_depth):
            feature_idx, information_gain = Tree(data).split_criteria()
            if(information_gain > 0.0):
                node.info_gain = information_gain
                node.feature = feature_idx
                node.depth = depth
                node.left = self.fit_node(data[np.where(data[:, feature_idx] == 1)], depth + 1)
                node.right = self.fit_node(data[np.where(data[:, feature_idx] != 1)], depth + 1)
        return node

    def predictions(self, x, root):
        """
        :param x: one example
        :param root: root node
        :return: predicted value (+1 or -1)
        """
        preds = np.empty(self.max_depth + 1)
        node = root
        for d in range(self.max_depth + 1):
            preds[d] = node.label
            if node.feature is not None:
                if x[node.feature] == 1:
                    node = node.left
                else:
                    node = node.right
        return preds

    def predict(self, xs, root):
        """
        :param xs: examples
        :param root: root node of the tree
        :return: predicted values
        """
        preds = np.empty((xs.shape[0], self.max_depth + 1))
        for ind in range(xs.shape[0]):
            preds[ind, :] = self.predictions(xs[ind, :], root)
        return preds

    def print_tree(self, root_node, header_label):
        """
        this function traverses decision tree by breadth first search
        :param root_node: root node
        :param header_label: label of the feature
        """
        q = queue.Queue()
        q.put(root_node)
        level = 0
        while(q.empty() is False):
            size = q.qsize()
            items = []
            feature_ids = []
            information_gains = []
            for idx in range(0, size):
                item = q.get()
                if item is not None and item.feature is not None:
                    items.append(header_label[item.feature])
                    feature_ids.append(item.feature)
                    information_gains.append(item.info_gain)
                    q.put(item.left)
                    q.put(item.right)
            print("level: ", level, "  ---  ", items)
            print("gain:", information_gains)
            print("feature id", feature_ids)
            level += 1

def main():

    # parse csv data
    train_data = Data("pa4_train_X.csv", "pa4_train_y.csv")
    val_data = Data("pa4_dev_X.csv", "pa4_dev_y.csv")

    xTrain = train_data.getX().values[:, :]
    yTrain = train_data.getY().apply(lambda d: -1 if d[0] == 0 else 1, axis=1).values.reshape(len(xTrain),1)

    xVal = val_data.getX().values[:, :]
    yVal = val_data.getY().apply(lambda d: -1 if d[0] == 0 else 1, axis=1).values.reshape(len(xVal),1)

    print(' ---------- Decision Tree by Yeojin Kim ---------- \n')

    train_accuracies =[]
    validate_accuracies = []
    for depth in [2, 5, 10, 20, 25, 30, 40, 50]:
        start_time = time.time()

        decision_tree = DecisionTree(depth)
        root_node = decision_tree.fit(xTrain, yTrain, 0)
        train_predict = decision_tree.predict(xTrain, root_node)[:, -1]
        val_predict = decision_tree.predict(xVal, root_node)[:, -1]
        end_time = time.time()

        decision_tree.print_tree(root_node, train_data.getHeaderLabel())

        train_accuracies.append(np.sum(np.where(yTrain.flatten() == train_predict, 1, 0))/yTrain.shape[0])
        validate_accuracies.append(np.sum(np.where(yVal.flatten() == val_predict, 1, 0))/yVal.shape[0])

        print("---------- depth: ", depth, "----------")
        print("train accuracy: ", np.sum(np.where(yTrain.flatten() == train_predict, 1, 0))/yTrain.shape[0])
        print("validation accuracy: ", np.sum(np.where(yVal.flatten() == val_predict, 1, 0))/yVal.shape[0])
        print("Time", end_time - start_time, "seconds")

    plt.plot([2, 5, 10, 20, 25, 30, 40, 50], train_accuracies, label='validation accuracy')
    plt.plot([2, 5, 10, 20, 25, 30, 40, 50], validate_accuracies, label='validation accuracy')
    plt.legend(["train data", "validate data"], loc=4)
    plt.xlabel('Tree Depth')
    plt.ylabel('Accuracy')
    plt.title('Train and Validation accuracy versus Tree Depth')
    plt.savefig('DecisionTree.png')
    plt.show()

if __name__ == '__main__':
    main()
