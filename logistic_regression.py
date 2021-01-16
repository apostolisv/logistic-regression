#Binomial logistic regression

import numpy as np
import matplotlib.pyplot as plt
from data import LoadData

lamba = 1


def sigmoid_function(theta, x):
    return 1.0/(1 + np.exp(-np.dot(theta.T, x)))


def train():
    pass


def test():
    pass


def cost():
    pass


def logistic_regression():
    pass



load = LoadData()


'''
load.createDictionary()
load.getVector('test', 'neg')
load.getVector('test', 'pos')
load.getVector('train', 'neg')
load.getVector('train', 'pos')
'''
