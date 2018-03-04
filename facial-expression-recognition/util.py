import numpy as np
import pandas as pd

from collections import defaultdict


def getData(balance_ones=False):
    # images are 48x48 = 2304 size vectors
    train_test = defaultdict(lambda: 0)
    Y = []
    X = []
    first = True
    for line in open('../large_files/fer2013/fer2013.csv'):
        if first:
            first = False
        else:
            row = line.split(',')
            # 'Training\n': 28709, 'PublicTest\n': 3589, 'PrivateTest\n': 3589
            if row[2] == 'Training\n':
                Y.append(int(row[0]))
                X.append([int(p) for p in row[1].split()])
            train_test[row[2]] += 1

    X, Y = np.array(X) / 255.0, np.array(Y)

    if balance_ones:
        # balance the 1 class
        X0, Y0 = X[Y != 1, :], Y[Y != 1]
        X1 = X[Y == 1, :]
        X1 = np.repeat(X1, 9, axis=0)
        X = np.vstack([X0, X1])
        Y = np.concatenate((Y0, [1] * len(X1)))

    print("train_test = [ {} ]".format(train_test))
    return X, Y


def getBinaryData(l0, l1):
    """
	l0 and l1  are distinct values between 0 and 6 inclusive
	l0 will be labeled 0, l1 will be labeled 1
	"""
    Y = []
    X = []
    first = True
    for line in open('../large_files/fer2013/fer2013.csv'):
        if first:
            first = False
        else:
            row = line.split(',')
            y = int(row[0])
            if y == l0:
                Y.append(0)
                X.append([int(p) for p in row[1].split()])
            if y == l1:
                Y.append(1)
                X.append([int(p) for p in row[1].split()])
    return np.array(X) / 255.0, np.array(Y)


def sigmoid(A):
    return 1 / (1 + np.exp(-A))


def sigmoid_cost(T, Y):
    return -(T * np.log(Y) + (1 - T) * np.log(1 - Y)).sum()


def error_rate(targets, predictions):
    return np.mean(targets != predictions)
