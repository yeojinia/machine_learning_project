"""
Filename: 	    kernelized_perceptron.py
Author:   	    Yeojin Kim
Date:     	    November 2020
Description:    This file is an implementation of kernelized vanilla perceptron.
In this file, polynomial kernel (order: 1~5) maps the features to a high dimensional space
where data become linearly separable and accuracy on training dataset and validation dataset is compared. 
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

class KernelPerceptron:
    """
    A class for kernelized vanilla perceptron learning
    """
    def __init__(self, X, y, valX, valy, max_iteration):
        """
        :param X: training examples
        :param y: training ground truth label (0/1)
        :param valX: validation examples
        :param valy: validation ground truth label(0/1)
        :param max_iteration: maximum iteration
        """
        self.X = np.array(X)
        self.y = np.array(y)
        self.valX = np.array(valX)
        self.valy = np.array(valy)
        self.max_iteration = max_iteration

    def polynomial_kernel(self, X, X_, pow):
        """
        :param X: training examples
        :param X_: new samples
        :param pow: polynomial degree (order) p
        :return: kernel function with degree pow
        """
        kernel = (np.dot(X, X_.T) + np.ones((len(X), len(X_))))**pow
        return kernel

    def online_fit(self, learning_rate, pow):
        """
        :param learning_rate: learning rate
        :param pow: polynomial degree
        :return: alpha, training accuracy, validation accuracy, duration
        """
        self.alpha = np.zeros((self.X.shape[0]))
        K = self.polynomial_kernel(self.X, self.X, pow)
        valK = self.polynomial_kernel(self.X, self.valX, pow)

        training_accuracies = []
        validation_accuracies = []
        self.y = self.y.flatten()
        self.valy = self.valy.flatten()
        accumulated_time = 0
        iteration_durations = []
        for _ in range(0, self.max_iteration):
            start_time = time.time()
            idx = 0
            for xi, yi in zip(self.X, self.y):
                u_ = np.dot(np.multiply(self.alpha.T, self.y), K[idx])
                if np.sign(u_ * yi) <= 0:
                    self.alpha[idx] = self.alpha[idx] +1
                idx = idx + 1
            duration = time.time() - start_time
            accumulated_time = accumulated_time + duration
            iteration_durations.append(accumulated_time)

            # accuracies for training dataset and validation dataset
            train_accuracy = self.accuracy(K, self.y, self.y, self.alpha)
            validation_accuracy = self.accuracy(valK, self.y, self.valy, self.alpha)
            training_accuracies.append(train_accuracy)
            validation_accuracies.append(validation_accuracy)

        return self.alpha, training_accuracies, validation_accuracies, iteration_durations

    def accuracy(self, K, y_train, y, alpha):
        y_predict = np.sign(np.dot(np.multiply(alpha, y_train), K))
        error = np.count_nonzero(y_predict - y)
        return 1.-(float)(error)/len(y)

class Data:

    def __init__(self, fileX, fileY):
        self.cols = list(pd.read_csv(fileX, nrows=1))
        self.X = pd.read_csv(fileX)
        self.y = pd.read_csv(fileY)

    def getX(self):
        return self.X

    def getY(self):
        return self.y

def main():

    train_data = Data("pa3_train_X.csv", "pa3_train_y.csv")
    validation_data = Data("pa3_dev_X.csv", "pa3_dev_y.csv")

    mean_of_train_accuracy = []
    mean_of_validate_accuracy = []
    std_of_train_accuracy = []
    std_of_validate_accuracy = []
    best_train_accuracy =[]
    best_validate_accuracy = []
    learning_rate = 1
    total_times = []
    num_iteration = 100
    for pow in [1,2,3,4,5]:

        print("power: ", pow)
        alpha, online_train_accuracy, online_validate_accuracy, times = KernelPerceptron(train_data.getX(), train_data.getY(), validation_data.getX(),
                                                                                  validation_data.getY(), max_iteration=num_iteration).online_fit(learning_rate = learning_rate, pow =pow)

        print(len(times))
        total_times.append(times)
        print(times[len(times)-1])

        mean_of_train_accuracy.append(round(np.mean(online_train_accuracy),3))
        mean_of_validate_accuracy.append(round(np.mean(online_validate_accuracy),3))
        std_of_train_accuracy.append(round(np.std(online_train_accuracy),3))
        std_of_validate_accuracy.append(round(np.std(online_validate_accuracy),3))
        best_train_accuracy.append(round(np.max(online_train_accuracy),3))
        best_validate_accuracy.append(round(np.max(online_validate_accuracy),3))

        plt.plot(list(range(1, num_iteration+1)), online_train_accuracy, label="training dataset")
        plt.plot(list(range(1, num_iteration+1)), online_validate_accuracy, label="validate dataset")
        plt.title("part2 online (p="+ str(pow) + ") training data")
        plt.ylabel('Accuracy')
        plt.xlabel('iteration')
        plt.legend(["training dataset", "validate dataset"], loc=4 )
        plt.savefig("part2_online_accuracy(p="+str(pow)+").png", format='png')
        plt.close()

    for idx in range(0, 5):
        plt.plot(list(range(1, num_iteration+1)), total_times[idx])
    plt.title("part2-a time measurement")
    plt.ylabel('time')
    plt.xlabel('iteration')
    plt.legend(["pow = 1", "pow = 2","pow = 3", "pow = 4", "pow = 5"], loc=4)
    plt.savefig("part2_time_measurement.png", format='png')
    plt.close()

    fig = go.Figure(data=[go.Table(header=dict(values=['', 'best_train_accuracy', 'best_validate_accuracy', 'mean_of_train_accuracy', 'mean_of_validate_accuracy',
                                                       'std_of_train_accuracy', 'std_of_validate_accuracy']),
                                   cells=dict(values=[['p=1', 'p=2', 'p=3', 'p=4', 'p=5'], best_train_accuracy, best_validate_accuracy, mean_of_train_accuracy, mean_of_validate_accuracy,
                                                      std_of_train_accuracy, std_of_validate_accuracy]))
                          ])
    fig.show()


if __name__ == '__main__':
    main()
