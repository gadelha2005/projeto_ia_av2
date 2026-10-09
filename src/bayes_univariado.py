import numpy as np


class BayesUnivariado:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.priori = np.array([(y == c).mean() for c in self.classes])
        self.medias = np.array([X[y == c].mean(axis=0) for c in self.classes])
        variancias = np.array([X[y == c].var(axis=0) for c in self.classes])
        self.variancias = variancias + 1e-9 * X.var(axis=0).max()
        return self

    def _log_posteriori(self, X):
        dif = X[:, None, :] - self.medias[None, :, :]
        log_p = -0.5 * np.log(2 * np.pi * self.variancias) - dif ** 2 / (2 * self.variancias)
        return np.log(self.priori) + log_p.sum(axis=2)

    def predict(self, X):
        return self.classes[self._log_posteriori(X).argmax(axis=1)]
