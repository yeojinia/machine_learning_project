"""
This file is an implementation of Vanilla Perceptron Classifier.
In this file, online perceptron accuracy and its averaged perceptron accuracy
are experimented on training dataset and validation dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def print_accuracy_chart(data1, data2, dataType):
    plt.plot(data1)
    plt.plot(data2)
    plt.title("(Accuracy - " + dataType + ")")
    plt.ylabel('Accuracy')
    plt.xlabel('Iteration')
    plt.legend(["Online","Average"], loc=4)
    plt.savefig("part1_" + dataType + "accuracy.png", format='png')
    plt.close()

def accuracy(w, X, y):
    predict = (y == np.sign(np.dot(w, np.transpose(X))))
    return (float)(np.count_nonzero(predict))/len(y)

class VanillaPerceptron:
    """
    Vanilla Perceptron Classifier
    """
    def __init__(self, X, y, valX, valy, max_iteration):
        """
        :param X: training data
        :param y: training ground truth (0/1)
        :param valX: validation data
        :param valy: validation ground truth (0/1)
        :param max_iteration: maximum iteration (user define)
        """
        self.X = np.array(X)
        self.y = np.array(y)
        self.valX = np.array(valX)
        self.valy = np.array(valy)
        self.max_iteration = max_iteration

    def fit(self):
        # w: weight vectors for online perceptron
        # w_hat : weight vectors for averaged perceptron
        self.w = np.zeros(self.X.shape[1])
        self.w_hat = np.zeros(self.X.shape[1])

        train_online_perceptron_accuracy = []
        train_average_perceptron_accuracy = []
        validate_online_perceptron_accuracy = []
        validate_average_perceptron_accuracy = []
        self.y = self.y.flatten()
        self.valy = self.valy.flatten()

        counter = 1
        for _ in range(0, self.max_iteration):
            idx = 0
            for xi, yi in zip(self.X, self.y):
                if yi * np.dot(self.w, xi.T) <= 0:
                    self.w = self.w + np.multiply(xi, yi)
                idx = idx + 1
                self.w_hat = np.add(np.multiply(counter, self.w_hat), self.w) / (counter + 1)
                counter = counter + 1

            train_online_perceptron_accuracy.append(accuracy(self.w, self.X, self.y))
            validate_online_perceptron_accuracy.append(accuracy(self.w, self.valX, self.valy))
            train_average_perceptron_accuracy.append(accuracy(self.w_hat, self.X, self.y))
            validate_average_perceptron_accuracy.append(accuracy(self.w_hat, self.valX, self.valy))

        return self.w.T, self.w_hat.T, train_online_perceptron_accuracy, train_average_perceptron_accuracy, validate_online_perceptron_accuracy, validate_average_perceptron_accuracy

class Data:

    def __init__(self, fileX, fileY):
        self.X = pd.read_csv(fileX)
        self.y = pd.read_csv(fileY)

    def getX(self):
        return self.X
    def getY(self):
        return self.y

def main():

    train_data = Data("pa3_train_X.csv", "pa3_train_y.csv")
    dev_data = Data("pa3_dev_X.csv", "pa3_dev_y.csv")

    w, w_hat, train_online_err, train_avg_err, val_online_err, val_avg_err = VanillaPerceptron(train_data.getX(), train_data.getY(), dev_data.getX(), dev_data.getY(), max_iteration=100).fit()

    print(val_avg_err)
    print("Accuracy: ", max(val_avg_err))
    print("Index of best validation accuracy: ", val_avg_err.index(max(val_avg_err)) + 1)

    # print out all data
    print_accuracy_chart(train_online_err, train_avg_err, "training data with iteration 100")
    print_accuracy_chart(val_online_err, val_avg_err, "validation data with iteration 100")

if __name__ == '__main__':
    main()
