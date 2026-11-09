import numpy as np


class PerceptronSimples:
    def __init__(self, taxa_aprendizado=0.01, epocas=50, semente=42):
        self.taxa = taxa_aprendizado
        self.epocas = epocas
        self.semente = semente

    def fit(self, X, y):
        rng = np.random.default_rng(self.semente)
        self.w = rng.uniform(-0.5, 0.5, X.shape[1])
        self.b = 0.0
        self.erros_por_epoca = []
        for _ in range(self.epocas):
            erros = 0
            for i in rng.permutation(len(X)):
                erro = y[i] - self._degrau(X[i] @ self.w + self.b)
                if erro != 0:
                    self.w += self.taxa * erro * X[i]
                    self.b += self.taxa * erro
                    erros += 1
            self.erros_por_epoca.append(erros)
        return self

    @staticmethod
    def _degrau(u):
        return np.where(u >= 0, 1, 0)

    def predict(self, X):
        return self._degrau(X @ self.w + self.b)
