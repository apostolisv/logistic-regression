#Binomial Logistic Regression

import random
import numpy as np
import matplotlib.pyplot as plt
from data import LoadData


def sigmoid_function(theta, x):
    return 1.0/(1 + np.exp(-np.dot(theta.T, x)))  #returns >= 0.5, if theta.T*x >= 0


def train():
    for i in range(m):
        pass


def test():
    pass


def jcost(theta, x, y):
    c = 0
    for i in range(1, m):
        c += cost(sigmoid_function(theta, x), y)
    return -c/m


def cost(h, y):
    return np.mean(-y * np.log(h) - (1-y) * np.log(1-h))


def stochastic_gradient_descent():
    random.shuffle(train0, 0.1)
    random.shuffle(train1, 0.1)
    i = 1
    s = 0
    pass


def logistic_regression():
    pass


def evaluate(train_list, y):
    if train_list[y] == train0[y]:
        return 0
    return 1

l = 1
data = LoadData()
data.createDictionary()
weights = {0, 0, 0, 0}
train1 = data.getVector('train', 'pos')
train0 = data.getVector('train', 'neg')
test1 = data.getVector('test', 'pos')
test0 = data.getVector('test', 'neg')
m = len(train0) + len(train1)



'''
xi = dianysma
yi = 0 h 1 / neg h pos
'''



