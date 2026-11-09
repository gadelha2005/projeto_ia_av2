import numpy as np


ATIVACOES = {
    "sigmoide": (
        lambda u: 1.0 / (1.0 + np.exp(-u)),
        lambda a: a * (1.0 - a),
    ),
    "tanh": (
        np.tanh,
        lambda a: 1.0 - a ** 2,
    ),
    "relu": (
        lambda u: np.maximum(0.0, u),
        lambda a: (a > 0).astype(float),
    ),
}


class MLPRegressor:
    def __init__(self, camadas_ocultas=(10,), ativacao="tanh", taxa_aprendizado=0.01,
                 epocas=100, tamanho_lote=32, semente=42):
        self.camadas_ocultas = tuple(camadas_ocultas)
        self.nome_ativacao = ativacao
        self.f, self.df = ATIVACOES[ativacao]
        self.taxa = taxa_aprendizado
        self.epocas = epocas
        self.lote = tamanho_lote
        self.semente = semente

    def _iniciar_pesos(self, n_entradas, rng):
        tamanhos = [n_entradas, *self.camadas_ocultas, 1]
        self.W, self.b = [], []
        for n_in, n_out in zip(tamanhos[:-1], tamanhos[1:]):
            escala = np.sqrt(2.0 / n_in) if self.nome_ativacao == "relu" else np.sqrt(1.0 / n_in)
            self.W.append(rng.normal(0.0, escala, (n_in, n_out)))
            self.b.append(np.zeros(n_out))

    def _forward(self, X):
        ativ = [X]
        for i in range(len(self.W)):
            u = ativ[-1] @ self.W[i] + self.b[i]
            ativ.append(u if i == len(self.W) - 1 else self.f(u))
        return ativ

    def _backward(self, ativ, y):
        n = len(y)
        delta = (ativ[-1] - y) * (2.0 / n)
        for i in reversed(range(len(self.W))):
            grad_W = ativ[i].T @ delta
            grad_b = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.W[i].T) * self.df(ativ[i])
            self.W[i] -= self.taxa * grad_W
            self.b[i] -= self.taxa * grad_b

    def fit(self, X, y):
        rng = np.random.default_rng(self.semente)
        self.y_media, self.y_desvio = y.mean(), y.std() or 1.0
        y_n = ((y - self.y_media) / self.y_desvio).reshape(-1, 1)
        self._iniciar_pesos(X.shape[1], rng)
        self.perda_por_epoca = []
        for _ in range(self.epocas):
            idx = rng.permutation(len(X))
            for ini in range(0, len(X), self.lote):
                lote = idx[ini:ini + self.lote]
                self._backward(self._forward(X[lote]), y_n[lote])
            self.perda_por_epoca.append(float(((self._forward(X)[-1] - y_n) ** 2).mean()))
        return self

    def predict(self, X):
        return self._forward(X)[-1].ravel() * self.y_desvio + self.y_media
