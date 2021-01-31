# Binomial Logistic Regression

import random
import numpy as np
import matplotlib.pyplot as plt
from data import LoadData
from scipy.special import expit

np.seterr(divide='ignore', invalid='ignore')    # ignore warnings caused by extreme values of random starting weights


# Global variables
l_value = [4]
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
split = [10, 20, 30, 40, 50, 60, 70, 80, 90]

m = 0
train1 = []
train0 = []
data_vector = []
test1 = []
test0 = []
weights = []
weights_history = []
dev_data = []
temp = []
data = LoadData()


def logistic_function(x, w):   # sigmoid function
    return expit(np.dot(w, x[:-1]))  # same as 1.0/(1 + np.exp(-np.dot(weights, x[:-1]))) without overflow warning
# last element of x[] = category (or y)


def train(train_total):        # train
    global train0, train1, m, data_vector, weights, dev_data, temp
    if len(temp) == 0:
        input('press ENTER to create dictionary')
        data.createDictionary()
        input('press ENTER to load training data')
        print('Loading positive train data...')
        train1.extend(data.getVector('train', 'pos'))
        print('Loading negative train data...')
        train0.extend(data.getVector('train', 'neg'))
        temp = train0 + train1
        random.shuffle(temp)
    dev_data = temp[int(len(temp) * train_total / 100):]   # train:dev split
    data_vector = temp[:int(len(temp) * train_total / 100)]
    m = len(data_vector)
    input('press ENTER to train')
    print('Calculating weights...')
    weights = np.zeros((len(data_vector[0]) - 1))
    sgd()
    print('selecting best performing model...')
    select_model()


def test():     # test
    global test1, test0, data_vector
    data_vector = []
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    input('press ENTER to load testing data')
    print('Loading positive testing data...')
    test1 = data.getVector('test', 'pos')
    print('Loading negative testing data...')
    test0 = data.getVector('test', 'neg')
    data_vector = test0 + test1
    random.shuffle(data_vector)
    input('press ENTER to test')
    for i in range(len(data_vector)):
        if logistic_function(data_vector[i], weights) >= 0.50:
            if evaluate(i) == 0:
                false_positives += 1
            else:
                true_positives += 1
        else:
            if evaluate(i) == 1:
                false_negatives += 1;
            else:
                true_negatives += 1
    pos_precision = true_positives/(true_positives+false_positives)
    pos_recall = true_positives/(true_positives+false_negatives)
    neg_precision = true_negatives/(true_negatives+false_negatives)
    neg_recall = true_negatives/(true_negatives+false_positives)
    precision = 0.5 * (pos_precision + neg_precision)
    recall = 0.5 * (pos_recall + neg_recall)
    print("Accuracy: ", (true_negatives + true_positives) / len(data_vector) * 100, "%")
    print("Precision: ", precision * 100, "%")
    print("Recall: ", recall * 100, "%")



def test_external(path):    # test external comment
    vector = data.get_external(path)
    vector.append(-1)
    prediction = logistic_function(vector, weights) * 100
    answer = 'Comment has a {c:.2f}% chance of being positive!'
    print(answer.format(c=prediction))


def cost(w, l_val):     # total cost
    total = 0.0
    reg_value = 0       # regularized cost
    for i in range(1, len(w)):
        reg_value += np.square(w[i])
    reg_value *= l_val/(2*m)
    for i in range(m):
        h = logistic_function(data_vector[i], w)
        y = evaluate(i)
        total += y * np.log(h) + (1 - y) * np.log(1 - h)
    return -total/m + reg_value


def single_cost(x, l_val):      # single cost
    reg_value = 0
    for i in range(1, len(weights)):
        reg_value += np.square(weights[i])
    reg_value *= l_val/(2*m)
    h = logistic_function(data_vector[x], weights)
    y = evaluate(x)
    return (-y * np.log(h) - (1-y) * np.log(1-h)) + reg_value


def sgd():            # Stochastic Gradient Descent
    global weights, data_vector, l_value, weights_history
    for l_val in l_value:
        iterations = 0
        max_iterations = 4
        cost_history = [-1, 0]
        a = 0.01  # learning rate
        for i in range(len(weights)):   # start with random weights
            weights[i] = random.uniform(-3.14, 3.14)
        while iterations < max_iterations and not converges(cost_history[0], cost_history[1]):
            iterations += 1
            random.shuffle(data_vector)  # shuffle on every iteration
            for i in range(m):
                cost_history[0] = cost_history[1]
                cost_history[1] += single_cost(i, l_val)
                predicted_value = logistic_function(data_vector[i], weights)
                y = evaluate(i)
                weights[0] -= a * (predicted_value - y) * data_vector[i][0]
                for k in range(1, len(weights)):   # update weights
                    weights[k] -= a * ((predicted_value - y) * data_vector[i][k] - l_val/m * weights[k])    # Regularization
        weights_history.append(weights)


def evaluate(pos):
    return data_vector[pos][-1]


def evaluate_dev(pos):
    return dev_data[pos][-1]


def select_model():     # Selecting model with the best performing lambda value
    global weights, weights_history
    weights_cost = []
    for i in weights_history:
        error_count = 0
        for j in range(len(dev_data)):
            if logistic_function(dev_data[j], i) >= 0.5 and evaluate_dev(j) == 0:
                error_count += 1
            elif logistic_function(dev_data[j], i) < 0.5 and evaluate_dev(j) == 1:
                error_count += 1
        weights_cost.append(error_count)
    weights = weights_history[weights_cost.index(min(weights_cost))]


def converges(old, new):
    if abs(new - old) < pow(10, -3):
        return True
    return False


def get_stats():
    train_accuracy = []
    dev_accuracy = []
    precision = []
    recall = []
    for value in split:
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        train(value)
        for i in range(m):
            if logistic_function(data_vector[i], weights) >= 0.5:
                if evaluate(i) == 0:
                    false_positives += 1
                else:
                    true_positives += 1
            else:
                if evaluate(i) == 1:
                    false_negatives += 1;
                else:
                    true_negatives += 1
        train_accuracy.append((true_negatives + true_positives) / len(data_vector))
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        for i in range(len(dev_data)):
            if logistic_function(dev_data[i], weights) >= 0.5:
                if evaluate_dev(i) == 0:
                    false_positives += 1
                else:
                    true_positives += 1
            else:
                if evaluate_dev(i) == 1:
                    false_negatives += 1;
                else:
                    true_negatives += 1
        dev_accuracy.append((true_negatives + true_positives) / len(dev_data))
    plt.plot(train_accuracy, label='train data')
    plt.plot(dev_accuracy, label='dev data')
    plt.title("Train/Dev Accuracy")
    plt.xlabel("Train data %")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    train(80)
    for threshold in thresholds:
        for i in range(len(dev_data)):
            if logistic_function(dev_data[i], weights) >= threshold:
                if evaluate_dev(i) == 0:
                    false_positives += 1
                else:
                    true_positives += 1
            else:
                if evaluate_dev(i) == 1:
                    false_negatives += 1
                else:
                    true_negatives += 1
        pos_precision = true_positives / (true_positives + false_positives)
        pos_recall = true_positives / (true_positives + false_negatives)
        neg_precision = true_negatives / (true_negatives + false_negatives)
        neg_recall = true_negatives / (true_negatives + false_positives)
        precision.append(0.5 * (pos_precision + neg_precision))
        recall.append(0.5 * (pos_recall + neg_recall))
    plt.plot(recall, precision)
    plt.title("Precision/Recall")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()


def logistic_regression():
    #train(100)
    #test()
    #get_stats()
    return

logistic_regression()
