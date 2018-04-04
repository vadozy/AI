import numpy as np

from sortedcontainers import SortedList
# Note: You can't use SortedDict because the key is distance
# if 2 close points are the same distance away, one will be overwritten


class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        """ same as "train", but because it is a lazy classifier,
        all it does is saves X and y
        """
        self.X = X
        self.y = y

    def predict(self, X):
        y = np.zeros(len(X))
        for i, x in enumerate(X):
            sl = SortedList()
            for j, xt in enumerate(self.X):  # xt - training point x
                diff = x - xt
                d = diff.dot(diff)
                if len(sl) < self.k:
                    sl.add((d, self.y[j]))
                else:
                    if d < sl[-1][0]:
                        del sl[-1]
                        sl.add((d, self.y[j]))

            votes = {}
            for _, v in sl:
                votes[v] = votes.get(v, 0) + 1
            max_votes = 0
            max_votes_class = -1
            for v, count in votes.items():
                if count > max_votes:
                    max_votes = count
                    max_votes_class = v
            y[i] = max_votes_class
        return y

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)
