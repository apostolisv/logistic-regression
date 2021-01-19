# Binomial Logistic Regression

import random
import numpy as np
from data import LoadData
from scipy.special import expit


# Global variables

l_value = 1.4
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
    dev_data = np.array([data_vector[int(len(data_vector) * 80 / 100):]])   # 90:10 train:dev split
    data_vector = data_vector[:int(len(data_vector) * 80 / 100)]
    m = len(data_vector)
    input('press ENTER to train')
    print('Calculating weights...')
    weights = np.zeros((len(data_vector[0]) - 1))
    sgd()
    '''for i in range(5):
        sgd()
    print('selecting best performing model...')
    select_model()'''



def test():
    global test1, test0, data_vector
    error_count = 0
    input('press ENTER to load testing data')
    print('Loading positive testing data...')
    test1 = data.getVector('test', 'pos')
    print('Loading negative testing data...')
    test0 = data.getVector('test', 'neg')
    data_vector = test0 + test1
    random.shuffle(data_vector)
    input('press ENTER to test')
    for i in range(len(data_vector)):
        if abs(logistic_function(data_vector[i], weights) - evaluate(i)) > 0.5:
            error_count += 1
    error_rate = "Error rate: {e:.2f}%"
    print(error_rate.format(e=error_count / len(data_vector) * 100))


def test_external():
    url = input('Enter .txt file path\n').strip()
    vector = data.get_external(url)
    vector.append(-1)
    prediction = logistic_function(vector, weights) * 100
    answer = 'Comment has a {c:.2f}% chance of being positive!'
    print(answer.format(c=prediction))


def cost(w):
    total = 0.0
    for i in range(m):
        h = logistic_function(data_vector[i], w)
        y = evaluate(i)
        total += y * np.log10(h) + (1 - y) * np.log10(1 - h)
    return -total/m


def sgd():            # Stochastic Gradient Descent
    global weights, data_vector, l_value, weights_history
    iterations = 0
    max_iterations = 20
    cost_history = [0, 0]
    a = 0.1  # learning rate
    for i in range(len(weights)):   # start with random weights
        weights[i] = random.uniform(-1.0, 1.0)
    cost_history[0] = cost(weights)
    cost_history[1] = cost_history[0] + 1
    while iterations < max_iterations and not converges(cost_history[0], cost_history[1]):
        iterations += 1
        cost_history[0] = cost_history[1]
        cost_history[1] = cost(weights)
        random.shuffle(data_vector)  # shuffle on every iteration
        predicted_value = logistic_function(data_vector[-1], weights)    # update weights based only on the last example
        y = evaluate(-1)
        weights[0] -= a * (predicted_value - y) * data_vector[-1][0]
        for k in range(1, len(weights)):   # update weights
            weights[k] -= a * (predicted_value - y) * data_vector[-1][k] + l_value / m * weights[k]  # Regularization
    weights_history.append(weights)


def evaluate(pos):
    return data_vector[pos][-1]


def select_model():
    global weights
    weights_cost = []
    for i in range(5):
        weights_cost.append(cost(weights_history[i]))
    weights = weights_history[weights_cost.index(min(weights_cost))]


def converges(old, new):
    if abs(new - old) < pow(10, -6):
        return True
    return False


def logistic_regression():
    train()
    #test()
    test_external()
    # C:\Users\Apostolis\Desktop\test.txt


logistic_regression()
