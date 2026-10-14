import numpy as np

class RegressaoLinearMultipla:
    def fit(self, X, y):
        A = np.column_stack([np.ones(len(X)), X])
        self.coef = np.linalg.lstsq(A, y, rcond=None)[0]
        return self

    def predict(self, X):
        A = np.column_stack([np.ones(len(X)), X])
        return A @ self.coef
