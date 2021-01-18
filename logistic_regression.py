# Binomial Logistic Regression

import random
import numpy as np
import os
from data import LoadData
from scipy.special import expit


def logistic_function(x):
    xtemp = np.array(x[:-1])        # last element = category [y]
    result = expit(-np.dot(weights, xtemp))  # same as 1.0/(1 + np.exp(-np.dot(weights.T, xtemp))) without overflow warning
    if result == 1.0:
        return 0.99       # avoid log10 error due to overflow
    elif result == 0.0:
        return 0.01
    return result


def train():
    global train0, train1, m, train_data, weights
    input('press ENTER to create dictionary')
    data.createDictionary()
    input('press ENTER to load training data')
    print('Loading positive train data...')
    train1.extend(data.getVector('train', 'pos'))
    print('Loading negative train data...')
    train0.extend(data.getVector('train', 'neg'))
    train_data = train0 + train1
    random.shuffle(train_data)
    m = len(train_data)
    input('press ENTER to train')
    print('Calculating weights...')
    weights = np.zeros(len(train_data[0])-1)
    sgd()


def test():
    global test1, test0, test_data
    input('press ENTER to load testing data')
    print('Loading test1 data...\n')
    test1 = data.getVector('test', 'pos')
    print('Loading test0 data...\n')
    test0 = data.getVector('test', 'neg')
    test_data = test0 + test1
    random.shuffle(test_data)
    input('press ENTER to test')
    '''
    TEST
    '''


def cost():
    total = 0.0
    for i in range(m):
        h = logistic_function(train_data[i])
        y = evaluate(i)
        total += y * np.log10(h) + (1 - y) * np.log10(1 - h)
    return -total/m


def sgd():            # Stochastic Gradient Descent
    global weights, train_data
    iterations = 0
    max_iterations = 400
    a = 0.01  # learning rate
    for i in range(len(weights)):   # start with random weights
        weights[i] = random.uniform(-0.2, 0.2)
    while iterations < max_iterations:
        print(cost())
        random.shuffle(train_data)  # shuffle on every iteration
        iterations += 1
        predicted_value = logistic_function(train_data[-1])
        y = evaluate(-1)
        for k in range(len(weights)):   # update weights
            weights[k] -= a * (predicted_value - y) * train_data[-1][k]


def evaluate(pos):
    return train_data[pos][-1]


def evaluate_test(pos):
    return test_data[pos][-1]


def logistic_regression():
    train()
    test()


# Global variables
m = 0
train1 = []
train0 = []
train_data = []
test1 = []
test0 = []
test_data = []
weights = []

data = LoadData()
logistic_regression()

'''
TO DO
NORMALISATION
TESTING
'''