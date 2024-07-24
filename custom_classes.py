import numpy as np

class StandardScalerScratch:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        return self

    def transform(self, X):
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class LabelEncoderScratch:
    def __init__(self):
        self.classes_ = {}

    def fit(self, X):
        for i, col in enumerate(X.T):
            self.classes_[i] = {label: idx for idx, label in enumerate(np.unique(col))}
        return self

    def transform(self, X):
        X_encoded = np.zeros(X.shape, dtype=int)
        for i, col in enumerate(X.T):
            X_encoded[:, i] = [self.classes_[i][label] for label in col]
        return X_encoded

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, num_iterations=5000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.y = y

        for i in range(self.num_iterations):
            self.update_weights()

    def update_weights(self):
        A = self.sigmoid(np.dot(self.X, self.W) + self.b)
        tmp = (A - self.y.T)
        tmp = np.reshape(tmp, self.m)
        dW = np.dot(self.X.T, tmp) / self.m
        db = np.sum(tmp) / self.m
        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db

    def predict(self, X):
        Z = self.sigmoid(np.dot(X, self.W) + self.b)
        y_pred = np.where(Z > 0.5, 1, 0)
        return y_pred
