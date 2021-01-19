# Binomial Logistic Regression

import random
import numpy as np
from data import LoadData
from scipy.special import expit


# Global variables

l_value = 314.0
m = 0
train1 = []
train0 = []
data_vector = []
test1 = []
test0 = []
weights = []
weights_history = []
dev_data = []
data = LoadData()


def logistic_function(x, w):
    return expit(np.dot(w, x[:-1]))  # same as 1.0/(1 + np.exp(-np.dot(weights.T, x[:-1]))) without overflow
# last element of x[] = category


def train():
    global train0, train1, m, data_vector, weights, dev_data
    input('press ENTER to create dictionary')
    data.createDictionary()
    input('press ENTER to load training data')
    print('Loading positive train data...')
    train1.extend(data.getVector('train', 'pos'))
    print('Loading negative train data...')
    train0.extend(data.getVector('train', 'neg'))
    data_vector = train0 + train1
    random.shuffle(data_vector)
    dev_data = data_vector[int(len(data_vector) * 60 / 100):]   # train:dev split
    data_vector = data_vector[:int(len(data_vector) * 60 / 100)]
    m = len(data_vector)
    input('press ENTER to train')
    print('Calculating weights...')
    for i in range(3):
        weights = np.zeros((len(data_vector[0]) - 1))
        sgd()
    print('selecting best performing model...')
    select_model()


def test():
    global test1, test0, data_vector
    data_vector = []
    error_count = 0
    input('press ENTER to load testing data')
    print('Loading positive testing data...')
    test1 = data.getVector('test', 'pos')
    print('Loading negative testing data...')
    test0 = data.getVector('test', 'neg')
    data_vector = test0 + test1
    random.shuffle(data_vector)
    input('press ENTER to test')
    for i in range(m):
        if abs(logistic_function(data_vector[i], weights) - evaluate(i)) > 0.5:
            error_count += 1
    error_rate = "Success rate: {e:.2f}%"
    print(error_rate.format(e=100.0-(error_count / len(data_vector) * 100)))


def test_external():
    path = input('Enter .txt file path\n').strip()
    vector = data.get_external(path)
    vector.append(-1)
    prediction = logistic_function(vector, weights) * 100
    answer = 'Comment has a {c:.2f}% chance of being positive!'
    print(answer.format(c=prediction))


def cost(w):
    total = 0.0
    reg_value = 0       # regularized cost
    for i in range(1, len(w)):
        reg_value += np.square(w[i])
    reg_value *= l_value/(2*m)
    for i in range(m):
        h = logistic_function(data_vector[i], w)
        y = evaluate(i)
        total += y * np.log10(h) + (1 - y) * np.log10(1 - h)
    total += reg_value
    return -total/m


def single_cost(x):
    reg_value = 0
    for i in range(1, len(weights)):
        reg_value += np.square(weights[i])
    reg_value *= l_value/(2*m)
    h = logistic_function(data_vector[x], weights)
    y = evaluate(x)
    return -(y * np.log10(h) + (1-y) * np.log10(1-h) + reg_value)


def sgd():            # Stochastic Gradient Descent
    global weights, data_vector, l_value, weights_history, l_value
    l_value = l_value/8
    iterations = 0
    max_iterations = 5
    cost_history = [-1, 0]
    a = 0.03  # learning rate
    for i in range(len(weights)):   # start with random weights
        weights[i] = random.uniform(-1.0, 1.0)
    while iterations < max_iterations and not converges(cost_history[0], cost_history[1]):
        iterations += 1
        print(cost(weights))
        random.shuffle(data_vector)  # shuffle on every iteration
        for i in range(m):
            cost_history[0] = cost_history[1]
            cost_history[1] += single_cost(i)
            predicted_value = logistic_function(data_vector[i], weights)
            y = evaluate(i)
            weights[0] -= a * (predicted_value - y) * data_vector[i][0]     # predict underfit [lambda value too high]
            for k in range(1, len(weights)):   # update weights
                weights[k] -= a * (predicted_value - y) * data_vector[i][k] + l_value / m * weights[k]  # Regularization
    if not underfit():
        weights_history.append(weights)
    else:
        print('Lambda value too high!')


def evaluate(pos):
    return data_vector[pos][-1]


def select_model():     # Selecting model with the most accurate lambda value
    global weights, weights_history
    weights_cost = []
    for i in weights_history:
        error_count = 0
        for j in range(len(dev_data)):
            if abs(logistic_function(dev_data[j], i) - evaluate(j)) > 0.5:
                error_count += 1
        weights_cost.append(error_count)
    weights = weights_history[weights_cost.index(min(weights_cost))]


def converges(old, new):
    if abs(new - old) < pow(10, -3):
        return True
    return False


def underfit():
    total = 0.0
    for i in range(m):
        total += logistic_function(data_vector[i], weights)
    if abs(total/m - weights[0]) < pow(10, -2):
        return True
    return False


def logistic_regression():
    train()
    test()
    #test_external()
    # C:\Users\Apostolis\Desktop\test.txt


logistic_regression()
