#Binomial Logistic Regression

import random
import sys
import numpy as np
import matplotlib.pyplot as plt
from data import LoadData


def sigmoid_function(x):
    return abs(0.5 + (1.0/(1 + np.exp(-np.dot(weights.T, x)))))  #returns >= 0.5, if theta.T*x >= 0 [probability that y = 1]


def train():
    for i in range(m/2):
        pass
    for i in range(m/2):
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


def sgd(train_list):            #Stochastic Gradient Descent
    convergeRate = sys.maxsize
    for i in range(len(weights)):   #start with random weights
        weights[i] = random.randint(0, 50)
    random.shuffle(train_list)
    i = 1
    s = 0
    s += jcost(weights, train_list[i], evaluate(train_list, i))
    update_weights()

def logistic_regression():
    pass


def evaluate(train_list, pos):
    if train_list[pos] == train0[pos]:
        return 0
    return 1


def update_weights(train_list):
    for i in range(len(weights)):
        for j in range(1, m):
            temp = sigmoid_function(train_list[j] - evaluate(train_list, j)) * train_list[j]
        weights[i] -= a * temp

data = LoadData()
data.createDictionary()

train1 = data.getVector('train', 'pos')
train0 = data.getVector('train', 'neg')
test1 = data.getVector('test', 'pos')
test0 = data.getVector('test', 'neg')
m = len(train0) + len(train1)
weights = len(train0[0]) * [0] #theta


'''
xi = dianysma
xj = leksi dianysmatos
yi = 0 h 1 / neg h pos
'''
