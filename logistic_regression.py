# Binomial Logistic Regression

import random
import numpy as np
from data import LoadData


def sigmoid_function(x):
    return 1.0/(1 + np.exp(-np.dot(weights.T, x)))  # returns >= 0.5, if theta.T*x >= 0 [probability that y = 1]


def train():
    global train0, train1, m
    print('Loading train1 data...\n')
    train1.extend(data.getVector('train', 'pos'))
    print('Loading train0 data...\n')
    train0.extend(data.getVector('train', 'neg'))
    m = 10


def test():
    global test1, test0
    print('Loading test1 data...\n')
    test1 = data.getVector('test', 'pos')
    print('Loading test0 data...\n')
    test0 = data.getVector('test', 'neg')


def j_cost(x, y):
    c = 0
    for i in range(m):
        c += cost(sigmoid_function(x), y)
    return c/m


def cost(h, y):
    return -y * np.log(h) - (1-y) * np.log(1-h)


def sgd(train_list):            # Stochastic Gradient Descent
    global weights
    iterations = 0
    max_iterations = 20
    s = 0
    h = 0.75  # learning rate
    for i in range(len(weights)):   # start with random weights
        weights[i] = random.randint(0, 50)
    random.shuffle(train_list)
    while iterations < max_iterations and converges(s, train_list):
        iterations += 1
        s = 0
        for i in range(len(train_list)):
            s += j_cost(train_list[i], evaluate(train_list, i))
            i += 1
            for k in range(len(weights)):   # update weights
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
            x += j_cost(train_list[i], evaluate(train_list, i))
        if x > s:
            return True
        else:
            return False


def total():
    total_cost = 0
    for i in range(len(train_data)):
        pass


data = LoadData()
data.createDictionary()


train1 = []
train0 = []
m = 0
test1 = []
test0 = []

weights = np.zeros(len(train0[0]))

train_data = train0 + train1
random.shuffle(train_data)

sgd(train_data)


'''
xi = vector
xj = vector element
yi = 0 h 1 / neg h pos
'''
