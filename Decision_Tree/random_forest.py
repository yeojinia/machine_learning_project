"""
Filename: 	    random_forest.py
Author:   	    Yeojin Kim
Date:     	    November 2020
Description:    This file is an implementation of Random Forest for binary classification.

"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

class Data:
    """
    A class for Data
    """
    def __init__(self, feature, classfied_y):
        self.X = pd.read_csv(feature)
        self.y = pd.read_csv(classfied_y, header=None)

    def getX(self):
        return self.X
    def getY(self):
        return self.y

class Node:
    """
    A calss for Node
    """
    def __init__(self):
        self.depth = None
        self.feature = None
        self.left = None
        self.right = None
        self.label = None

class Tree:
    """
    A class for Tree
    """
    def __init__(self, data):
        self.data = data

    def get_entropy(self, count, total):
        """
        :param count: count
        :param total: totals
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
        parent_uncertainty = self.get_entropy(plus, plus+ minus) + self.get_entropy(minus, plus + minus)
        # x[feature_idx] == 1 (1)
        left_uncertainty = self.get_entropy(left_plus, left_plus + left_minus) + self.get_entropy(left_minus, left_plus + left_minus)
        # x[feature_idx] == 0 (-1)
        right_uncertainty = self.get_entropy(right_plus, right_plus + right_minus) + self.get_entropy(right_minus, right_plus + right_minus)
        left_prob = (left_plus + left_minus)/(plus + minus)
        right_prob = (right_plus + right_minus) / (plus + minus)
        gain = left_prob*left_uncertainty + right_prob*right_uncertainty
        return (parent_uncertainty - gain)

    def split_criteria(self, data):
        """
        :param data:
        :return: best feature index and its information gain
        """
        plus = np.sum(data[data[:, -1] >=0 ][:,-1])
        minus = -np.sum(data[data[:, -1] <0 ][:,-1])

        information_gain = -1
        feature_idx = 0
        for idx in range(0, data.shape[1]-1):
            left_plus = 0
            left_minus = 0
            right_plus = 0
            right_minus = 0
            XYs = np.copy(data[:, [idx, -1]])
            for _xy in XYs:
                if(_xy[0] == 1 and _xy[1] > 0):
                    left_plus += _xy[1]
                elif(_xy[0] == 1 and _xy[1] <=0):
                    left_minus -= _xy[1]
                elif(_xy[0] == 0 and _xy[1] >0):
                    right_plus += _xy[1]
                elif(_xy[0] == 0 and _xy[1] <=0):
                    right_minus -= _xy[1]
            if(plus + minus != 0):
                gain = self.information_gain(plus, minus, left_plus, left_minus, right_plus, right_minus)
                if(gain > information_gain):
                    feature_idx = idx
                    information_gain = gain
        return feature_idx, information_gain

class RandomForest:
    """
    A class for Random Forest
    """
    def __init__(self, T, m, dmax):
        """
        :param T: Tree Size
        :param m: the number of features
        :param dmax: the depth of tree
        """
        self.T = T
        self.m = m
        self.dmax = dmax
        self.forest = []

    def fit(self, x, y, depth=0):
        """
        :param x: examples
        :param y: ground truth (-1, 1)
        :param depth: starting depth
        :return: forest
        """
        X = np.array(x)
        Y = np.array(y)
        data = np.hstack((X, Y))
        np.random.seed(1)
        for tree in range(0, self.T):
            # make new bootstrapped dataset, it allows duplicate entries in the bootstrapped dataset.
            cur_data = np.copy(data[np.random.choice(range(X.shape[0]), X.shape[0], replace=True), :])
            self.forest.append(self.fit_node(cur_data, 0))
        return self.forest

    def fit_node(self, data, depth):
        """
        :param data: x+y
        :param depth: current depth
        :return: current node
        """
        plus = np.sum(data[data[:, -1] >= 0][:, -1])
        minus = -np.sum(data[data[:, -1] < 0][:, -1])
        node = Node()
        node.depth = depth
        if (plus > minus):
            node.label = 1
        else:
            node.label = -1

        if (depth <= self.dmax):
            # consider a random a subset of variables at each step, i.e., randomly select m variables as candidates
            bagged_features = np.random.choice(list(range(data.shape[1] - 1)), self.m, replace=False)
            bagged_features = np.hstack((bagged_features, np.array([-1])))
            feature_idx, information_gain = Tree(data).split_criteria(data[:, bagged_features])
            if (information_gain > 0.0):
                node.feature = int(bagged_features[feature_idx])
                node.depth = depth
                node.left = self.fit_node(data[np.where(data[:, node.feature] == 1)], depth + 1)
                node.right = self.fit_node(data[np.where(data[:, node.feature] != 1)], depth + 1)
        return node

    def predictions(self, x):
        """
        get all predictions of each trees for a example 'x'
        :param x: one example
        :return: prediction result
        """
        preds = np.zeros((self.T, 1))
        plus = 0
        minus = 0
        for tree_idx, tree in enumerate(self.forest):
            node = tree
            pred = node.label
            while node.feature is not None:
                if x[node.feature] == 1:
                    node = node.left
                else:
                    node = node.right
                pred = node.label
            if pred == 1:
                plus += 1
            else:
                minus += 1
            # check which option received more votes
            preds[tree_idx] = 1 if plus > minus else -1
        return preds

    def predict(self, xs):
        """
        run the data down all of the trees in the random forest
        :param xs: examples
        :return: predictions over forest
        """
        preds = np.zeros((xs.shape[0], self.T))
        for idx in range(xs.shape[0]):
            preds[idx, :] = self.predictions(xs[idx, :]).reshape(preds[idx, :].shape)
        return preds

def main():

    train_data = Data("./../pa4_train_X.csv", "./../pa4_train_y.csv")
    val_data = Data("./../pa4_dev_X.csv", "./../pa4_dev_y.csv")
    xTrain = train_data.getX().values[:, :]
    yTrain = train_data.getY().apply(lambda d: -1 if d[0] == 0 else 1, axis=1).values.reshape(len(xTrain),1)

    xVal = val_data.getX().values[:, :]
    yVal = val_data.getY().apply(lambda d: -1 if d[0] == 0 else 1, axis=1).values.reshape(len(xVal),1)

    print(' ---------- Part 2 Random Forest ---------- \n')

    for index, dmax in enumerate([2, 10, 25]):

        train_accuracies = []
        validate_accuracies = []

        num_trees = list(range(10, 101, 10))

        for m in [5, 25, 50, 100]:
            start_time = time.time()
            random_forest = RandomForest(num_trees[-1], m, dmax)
            random_forest.fit(xTrain, yTrain, 0)
            end_time = time.time()
            print('[depth:', str(m),']  Time to train:', end_time - start_time, 'sec')

            start_time = time.time()
            train_preds = random_forest.predict(xTrain)
            val_preds = random_forest.predict(xVal)
            end_time = time.time()
            print('[depth:', str(m),']  Time to predict:', end_time - start_time, 'sec')
            accuracies = np.zeros((len(num_trees), 2))

            # get accuracies over tree size (10, 20, 30 ... , 100)
            for ind, tree_count in enumerate(num_trees):
                accuracies[ind, 0] = np.mean(yTrain == train_preds[:, num_trees[ind] - 1].reshape(yTrain.shape)) * 100
                accuracies[ind, 1] = np.mean(yVal == val_preds[:, num_trees[ind] - 1].reshape(yVal.shape)) * 100
            train_accuracies.append(accuracies[:, 0])
            validate_accuracies.append(accuracies[:, 1])

            columns = ['Train', 'Validation']
            accuracy_pd = pd.DataFrame(data=accuracies, index=num_trees, columns=columns)
            print('Accuracies')
            print(accuracy_pd)


        for m, accuracies in zip([5, 25, 50, 100],train_accuracies):
            plt.plot(num_trees, accuracies, label='train accuracy')
        plt.legend(["m = 5 ", "m = 25 ", " m= 50", " m = 100"], loc=4)
        plt.xticks(num_trees)
        plt.xlabel('Tree Size')
        plt.ylabel('Accuracy')
        plt.title('Train accuracy versus number of trees with depth %d' % dmax)
        plt.savefig('random_forest_dmax'+str(dmax)+'train.png')
        plt.gcf().clear()

        for m, accuracies in zip([5, 25, 50, 100],validate_accuracies):
            plt.plot(num_trees, accuracies, label='validation accuracy')
        plt.legend(["m = 5 ", "m = 25 ", " m= 50", " m = 100"], loc=4)
        plt.xticks(num_trees)
        plt.xlabel('Tree Size')
        plt.ylabel('Accuracy')
        plt.title('Validation accuracy versus number of trees with depth %d' % dmax)
        plt.savefig('random_forest_dmax'+str(dmax)+'validate.png')
        plt.gcf().clear()


if __name__ == '__main__':
    main()