import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt

numerical_features = ["year","month", "day", "bedrooms", "bathrooms","sqft_living","sqft_lot","floors","waterfront","view","condition","grade","sqft_above","sqft_basement","yr_built","yr_renovated","zipcode","lat","long","sqft_living15","sqft_lot15"]
categorical_features = ["waterfront","condition","grade"]

class Linear_Regression:

    def __init__ (self, x, y):
        """
        :param x: independent variables in linear regression model
        :param y: dependent variable in linear regression model
        """
        self.x = x
        self.y = y

    def train(self, max_iteration, learning_rate, epsilon):
        """
        :param max_iteration: int, max iteration
        :param learning_rate: float, learning rate
        :param epsilon: float, epsilon for checking convergence
        :return: weights of the model, mean square error cost, number of iterations until convergence
        """
        self.w_ = np.zeros((self.x.shape[1], 1))
        N = self.x.shape[0]
        np.savetxt("foo.csv", self.x, delimiter=",")
        self.cost_ = []
        iteration = 0
        idx = 0

        # Gradient Descent for MSE
        for idx in range(0, max_iteration):
            idx = idx+1
            iteration = idx
            y_pred = np.dot(self.x, self.w_)
            residuals = y_pred - self.y
            gradient_vector = (2*np.dot(self.x.T, residuals))/N

            self.w_ -= ((learning_rate)*(gradient_vector))
            # objective is to minimize mean squared error
            cost = np.sum((residuals**2))/(N)
            self.cost_.append(cost)

            # convergence criterion
            if(np.linalg.norm(gradient_vector) < epsilon) :
                iteration = idx
                break

        return self.w_, self.cost_, iteration

    def get_MSE(self, x, y):
        """
        :param x: independent variables
        :param y: dependent variable
        :return: error between ground truth y and predicted y
        """
        N = x.shape[0]
        y_pred = np.dot(x, self.w_)
        residuals = (y_pred - y)
        cost = np.sum((residuals**2))/(N)
        return cost


class DataProcessing:

    def __init__(self, fileName):
        """
        :param fileName: string
        """
        # Read column names from file
        self.fileName = fileName
        self.cols = list(pd.read_csv(fileName, nrows=1))
        self.x = pd.read_csv(self.fileName, usecols=[i for i in self.cols if i != 'id' and i!= 'price'])
        self.y = pd.read_csv(self.fileName, usecols=[i for i in self.cols if i == 'price'])

    def normalize(self):
        """
        :return: feature normalization by the difference between min value and max value
        """
        for feature_names in self.x:
            max = (self.x[feature_names].max())
            min = (self.x[feature_names].min())
            if(max != min): # not divided by zero
                self.x[feature_names] = ((self.x[feature_names] - min)/(max - min))
        self.normalized_x = self.x
        return self.normalized_x

    def cleaning(self):
        """
        :return: pre-processing for date data
        """
        df = pd.DataFrame(data=self.x).drop(['date'], axis = 1)
        df['year'] = pd.DatetimeIndex(self.x['date']).year
        df['month'] = pd.DatetimeIndex(self.x['date']).month
        df['day'] = pd.DatetimeIndex(self.x['date']).day
        self.x = df
        return self.x

    def get_normalized_x(self):
        """
        :return: get normalized features
        """
        return self.normalized_x

    def get_x(self):
        """
        :return: get features
        """
        return self.x

    def get_y(self):
        """
        :return: get ground truth y
        """
        return self.y

def main():

    # learning rate (or step size) defines how far you travel.
    learning_rates = [10**-1, 10**-2, 10**-3, 10**-4, 10**-5, 10**-6, 10**-7]

    train_data = DataProcessing("PA1_train.csv")
    train_data.cleaning()
    train_data.normalize()

    validation_data = DataProcessing("PA1_dev.csv")
    validation_data.cleaning()
    validation_data.normalize()

    for learning_rate in learning_rates:
        model = Linear_Regression(train_data.get_normalized_x().to_numpy(), train_data.get_y().to_numpy())
        weights, cost, iteration = model.train(max_iteration=1000, learning_rate=learning_rate, epsilon=0.5)

        if (np.any(np.isnan(weights)) == False):
            training_mse = model.get_MSE(train_data.get_x(), train_data.get_y())
            validation_mse = model.get_MSE(validation_data.get_x(), validation_data.get_y())
            print("(part1) (TRAINING DATA) MSE with learning rate (" + str(learning_rate) + ") = ", training_mse["price"], ", Iteration =", iteration, "times")
            print("(part1) (VALIDATION DATA) MSE with learning rate (" + str(learning_rate) + ") = ", validation_mse["price"], "Iteration =", iteration, "times")
        else:
            print("(part1) (TRAINING DATA) MSE with learning rate (" + str(learning_rate) + ") = NaN")
            print("(part1) (VALIDATION DATA) MSE with learning rate (" + str(learning_rate) + ") = NaN")

        fig, ax = plt.subplots()
        plt.plot(cost)
        plt.title("(part1) MSE with learning rate = " + str(learning_rate) )
        plt.ylabel('MSE')
        plt.xlabel('epoch')
        plt.savefig("part1_" + str(learning_rate)+".png", format='png')

if __name__ == '__main__':
    main()
