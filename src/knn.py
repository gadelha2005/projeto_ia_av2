import numpy as np


def distancia_euclidiana(X_treino, x):
    return np.sqrt(((X_treino - x) ** 2).sum(axis=1))


def distancia_manhattan(X_treino, x):
    return np.abs(X_treino - x).sum(axis=1)


DISTANCIAS = {
    "euclidiana": distancia_euclidiana,
    "manhattan": distancia_manhattan,
}


class KNN:
    def __init__(self, k=5, distancia="euclidiana", tarefa="classificacao"):
        self.k = k
        self.dist = DISTANCIAS[distancia]
        self.tarefa = tarefa

    def fit(self, X, y):
        self.X = X
        self.y = y
        return self

    def _prever_um(self, x):
        d = self.dist(self.X, x)
        vizinhos = np.argpartition(d, self.k - 1)[: self.k]
        alvos = self.y[vizinhos]
        if self.tarefa == "regressao":
            return alvos.mean()
        valores, contagens = np.unique(alvos, return_counts=True)
        return valores[contagens.argmax()]

    def predict(self, X):
        return np.array([self._prever_um(x) for x in X])
