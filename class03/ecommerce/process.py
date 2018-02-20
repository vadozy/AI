import numpy as np
import pandas as pd


def get_data():
    df = pd.read_csv('ecommerce_data.csv')
    data = df.as_matrix()

    X = data[:, :-1]
    Y = data[:, -1]

    # Normalize numeric columns
    X[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
    X[:, 2] = (X[:, 2] - X[:, 2].mean()) / X[:, 2].std()

    # time of day X[:, 4] (or X[:, -1]) - categorical value: 0, 1, 2, 3
    N, D = X.shape
    X2 = np.zeros((N, D + 3))
    X2[:, 0:(D - 1)] = X[:, 0:(D - 1)]

    # one hot encoding for the last column 4 values
    for n in range(N):
        t = int(X[n, D - 1])
        X2[n, t + D - 1] = 1

    # # another way to hot encode
    # Z = np.zeros((N, 4))
    # Z[np.arange(N), X[:, D-1].astype(np.int32)] = 1
    # # X2[:, -4:] = Z
    # assert(np.abs(X2[:, -4:] - Z), sum() < 1e-10)

    return X2, Y

# take only classes of 0 or  (binary classification)


def get_binary_data():
    X, Y = get_data()
    X2 = X[Y <= 1]
    Y2 = Y[Y <= 1]
    return X2, Y2
