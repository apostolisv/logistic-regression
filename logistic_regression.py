#Binomial Logistic Regression

import random
import numpy as np
import matplotlib.pyplot as plt
from data import LoadData


def sigmoid_function(x):
    return 1.0/(1 + np.exp(-np.dot(weights.T, x)))  #returns >= 0.5, if theta.T*x >= 0 [probability that y = 1]


def test():
    pass


def jcost(x, y):
    c = 0
    for i in range(m):
        c += cost(sigmoid_function(x), y)
    return c/m


def cost(h, y):
    return -y * np.log(h) - (1-y) * np.log(1-h)


def sgd(train_list):            #Stochastic Gradient Descent
    iterations = 0
    max_iterations = 20
    s = 0
    h = 0.75 #learning rate
    for i in range(len(weights)):   #start with random weights
        weights[i] = random.randint(0, 50)
    random.shuffle(train_list)
    while iterations < max_iterations and converges(s, train_list):
        iterations += 1
        s = 0
        for i in range(len(train_list)):
            s += jcost(train_list[i], evaluate(train_list, i))
            i += 1
            for k in range(len(weights)):   #update weights
                weight_update = 0
                for j in range(m):
                    weight_update += sigmoid_function(train_list[j] - evaluate(train_list, j)) * train_list[j][k]
                weights[k] -= h * weight_update



def evaluate(train_list, pos):
    if train_list[pos] == train0[pos]:
        return 0
    return 1


def converges(s, train_list):
    if s == 0:
        return True
    else:
        x = 0
        for i in range(len(train_list)):
            x += jcost(train_list[i], evaluate(train_list, i))
        if x > s:
            return True
        else:
            return False


def total():
    total_cost = 0
    for i in range(len(traindata)):
        pass

data = LoadData()
data.createDictionary()

print('Loading train1 data...\n')
train1 = data.getVector('train', 'pos')
print('Loading train0 data...\n')
train0 = data.getVector('train', 'neg')
print('Loading test1 data...\n')
test1 = data.getVector('test', 'pos')
print('Loading test0 data...\n')
test0 = data.getVector('test', 'neg')
m = len(train0) + len(train1)
weights = len(train0[0]) * [0]

traindata = train0 + train1
random.shuffle(traindata)

sgd(traindata)


'''
xi = dianysma
xj = leksi dianysmatos
yi = 0 h 1 / neg h pos
'''
