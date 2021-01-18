# Binomial Logistic Regression

import random
import numpy as np
from data import LoadData
from scipy.special import expit


# Global variables

l_value = 0.4

m = 0
train1 = []
train0 = []
data_vector = []
test1 = []
test0 = []
weights = []
dev_data = []
data = LoadData()


def logistic_function(x):
    return expit(np.dot(weights, x[:-1]))  # same as 1.0/(1 + np.exp(-np.dot(weights.T, x[:-1]))) without overflow
# last element of x[] = category


def train():
    global train0, train1, m, data_vector, weights, dev_data
    input('press ENTER to create dictionary\n')
    data.createDictionary()
    input('press ENTER to load training data\n')
    print('Loading positive train data...')
    train1.extend(data.getVector('train', 'pos'))
    print('Loading negative train data...')
    train0.extend(data.getVector('train', 'neg'))
    data_vector = train0 + train1
    random.shuffle(data_vector)
    dev_data = np.array([data_vector[int(len(data_vector) * 90 / 100):]])   # 90:10 dev:train split
    data_vector = data_vector[:int(len(data_vector) * 90 / 100)]
    m = len(data_vector)
    input('press ENTER to train')
    print('Calculating weights...')
    weights = np.zeros((len(data_vector[0]) - 1))
    sgd()
    select_model()


def test():
    global test1, test0, data_vector
    error_count = 0
    input('press ENTER to load testing data\n')
    print('Loading test1 data...\n')
    test1 = data.getVector('test', 'pos')
    print('Loading test0 data...\n')
    test0 = data.getVector('test', 'neg')
    data_vector = test0 + test1
    random.shuffle(data_vector)
    input('press ENTER to test')
    for i in range(len(data_vector)):
        error_count += abs((logistic_function(data_vector[i]) + 0.5) + evaluate(i))
        print(str(logistic_function(data_vector[i])) + " - " + str(evaluate(i)))
    print('Error rate: ' + str(error_count / len(data_vector)))


def test_external():
    url = input('Enter .txt file path\n').strip()
    vector = data.getExternal(url)
    vector.append(-1)
    prediction = logistic_function(vector) * 100
    answer = 'Comment has a {c:.2f}% chance of being positive!'
    print(answer.format(c=prediction))


def cost():
    total = 0.0
    for i in range(m):
        h = logistic_function(data_vector[i])
        y = evaluate(i)
        total += y * np.log10(h) + (1 - y) * np.log10(1 - h)
    return -total/m


def sgd():            # Stochastic Gradient Descent
    global weights, data_vector, l_value
    iterations = 0
    max_iterations = 100
    a = 0.01  # learning rate
    for i in range(len(weights)):   # start with random weights
        weights[i] = random.uniform(-3.0, 3.0)
    while iterations < max_iterations:
        random.shuffle(data_vector)  # shuffle on every iteration
        iterations += 1
        predicted_value = logistic_function(data_vector[-1])
        #print(cost())
        y = evaluate(-1)
        weights[0] -= a * (predicted_value - y) * data_vector[-1][0]
        for k in range(1, len(weights)):   # update weights
            weights[k] -= a * (predicted_value - y) * data_vector[-1][k] + l_value / m * weights[k]  # Regularization


def evaluate(pos):
    return data_vector[pos][-1]


def select_model():
    pass


def logistic_regression():
    train()
    test_external()


logistic_regression()

'''
TO DO
NORMALISATION
TESTING
dev data
'''