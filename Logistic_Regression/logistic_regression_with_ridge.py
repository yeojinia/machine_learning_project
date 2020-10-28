# Adding independent variables to a logistic regression can lead to the model overfitting.
# In order to avoid overfitting on logistic regression, Ridge(L2) regularizer is applied,
# and its effect of the amount of regularization is experimented.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import count_nonzero

def sort_by_absolute(tup):
    return (sorted(tup, key=lambda x: abs(x[0])))

def predict(features, weights):
    """
    :param features:
    :param weights:
    :return:
    """
    z = np.dot(features, weights)
    return 1 / (1 + np.exp(-z))

def classify(probabilities):
    """
    :param probabilities:
    :return:
    """
    probable = np.zeros((probabilities.shape[0], 1))
    for idx in range(0, len(probabilities)):
        if (probabilities[idx] >= .5):
            probable[idx] = 1
        else:
            probable[idx] = 0
    return probable

def accuracy(predicted_labels, ground_truth_labels):
    """
    :param predicted_labels:
    :param actual_labels:
    :return:
    """
    diff = predicted_labels - ground_truth_labels
    return 1.0 - (float(np.count_nonzero(diff))/len(diff))

class Logistic_Regression:

    def __init__ (self, x, y):
        """
        :param x: float pd df
        :param y: float pd df
        """
        self.x = x
        self.y = y

    def sigmoid(self, scores):
        """
        :param scores:
        :return:
        """
        return 1 / (1 + np.exp(-scores))

    def cost_function(self, y, pred):
        c1_cost = - y * np.log(pred)
        c0_cost = (1 - y) * np.log(1 - pred)
        cost = c1_cost - c0_cost
        cost = cost.sum() / len(y)
        return cost

    def train_with_L2_reg(self, max_iteration, learning_rate, epsilon, lambda_):
        self.w_ = np.zeros((self.x.shape[1], 1))
        N = self.x.shape[0]
        self.cost_ = []

        iteration = 0

        for idx in range(0, max_iteration):
            iteration = idx + 1
            scores_ = np.dot(self.x, self.w_)
            y_pred = self.sigmoid(scores_)
            residuals = y_pred - self.y
            gradient_vector = (np.dot(self.x.T, residuals)) / N
            self.w_ -= ((learning_rate) * (gradient_vector))
            self.w_[1:] -= (learning_rate*lambda_*self.w_[1:])
            cost = self.cost_function(self.y, y_pred)
            self.cost_.append(cost)

        return self.w_, self.cost_, iteration

class DataProcessing:

    def __init__(self, fileX, fileY):
        """
        Initialize data structure here.
        :type size: int
        """
        # Read column names from file
        self.fileX = fileX
        self.fileY = fileY
        self.cols = list(pd.read_csv(fileX, nrows=1))
        self.x = pd.read_csv(self.fileX)
        self.y = pd.read_csv(self.fileY)

    def normalize(self):
        """
        :return:
        """
        self.name = []
        for feature in self.x:
            self.name.append(feature)
            min = self.x[feature].min()
            max = self.x[feature].max()
            if(max != min): # not divided by zero
                self.x[feature] = ((self.x[feature] - min)/(max - min))

    def getX(self):
        """
        :return:
        """
        return self.x

    def getY(self):
        """
        :return:
        """
        return self.y

    def getName(self):
        return self.name

def main():

    learning_rate_ = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    lambda_ = [10 ** 0, 10 ** -1, 10 ** -2, 10 ** -3, 10 ** -4, 10 ** -5]
    x = [_ for _ in range(len(lambda_))]

    for learning_rate in learning_rate_:
        train_data = DataProcessing("pa2_train_X.csv", "pa2_train_y.csv")
        train_data.normalize()

        validation_data = DataProcessing("pa2_dev_X.csv", "pa2_dev_y.csv")
        validation_data.normalize()

        training_accuracy = []
        validation_accuracy = []
        weights = []
        weight_result = []
        name = train_data.getName()

        for hyperparameter in lambda_:
            w, cost, iteration = Logistic_Regression(train_data.getX(), train_data.getY()).train_with_L2_reg(max_iteration=10000, learning_rate=learning_rate, epsilon=0.5, lambda_=hyperparameter)

            result_ = zip(w, name)
            weight_result.append(result_)
            weights.append(w)

            train_probabilities = predict(train_data.getX(), w)
            train_classificiations = classify(train_probabilities)
            train_acc = accuracy(train_classificiations, train_data.getY())
            training_accuracy.append(train_acc)
            print("Learning rate: ", learning_rate,  ", Hyperparameter lambda:", hyperparameter, ", Accuracy for Training: ", train_acc)

            validation_probabilities = predict(validation_data.getX(), w)
            validation_classificiations = classify(validation_probabilities)
            validation_acc = accuracy(validation_classificiations, validation_data.getY())
            validation_accuracy.append(validation_acc)
            print("Learning rate: ", learning_rate,  ", Hyperparameter lambda:", hyperparameter, ", Accuracy for Validation:", validation_acc)

        fig, ax = plt.subplots()
        plt.plot(training_accuracy)
        plt.xticks(x, lambda_, rotation='horizontal')
        plt.title("(part1) Data Accuracy (Training) with L2 regularization (α= " + str(learning_rate) +" )")
        plt.ylabel('Accuracy')
        plt.xlabel('Lambda(hyperparameter)')
        plt.savefig("LR_" + str(learning_rate) + "part1_training_data_accuracy.png", format='png')
        plt.close()

        fig2, ax2 = plt.subplots()
        plt.plot(validation_accuracy)
        plt.xticks(x, lambda_, rotation='horizontal')
        plt.title("(part1) Data Accuracy (Validation) with L2 regularization (α= " + str(learning_rate) +" )")
        plt.ylabel('Accuracy')
        plt.xlabel('Lambda(hyperparameter)')
        plt.savefig("LR_" + str(learning_rate) + "part1_validation_data_accuracy.png", format='png')
        plt.close()

        print("LR_" + str(learning_rate)+ "_training accuracy")
        print(training_accuracy)

        print("LR_" + str(learning_rate)+ "_validation accuracy")
        print(validation_accuracy)

        print("LR_" + str(learning_rate)+ "_weights")
        for lamb, weight, weight_result_ in zip(lambda_, weights, weight_result):
            print(lamb)
            print(sort_by_absolute(weight_result_))
            print("sparsity is " + str(1. - (count_nonzero(weight)/len(weight))), ", the number of feature w_j=0: ", str(len(weight) - count_nonzero(weight)))

if __name__ == '__main__':
    main()
