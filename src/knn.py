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
        return valores[contagens.argmax()]  # empate: menor rótulo

    def predict(self, X):
        return np.array([self._prever_um(x) for x in X])


def testar_knn():
    from carregar_dados import ler_arff
    from normalizacao import ZScore
    from validacao_cruzada import gerar_folds

    Xt = np.array([[0.0, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0]])
    yt = np.array([0, 0, 1, 1])
    for nome in DISTANCIAS:
        m = KNN(k=3, distancia=nome).fit(Xt, yt)
        print(f"toy {nome}:", m.predict(np.array([[0.5, 0.5], [10.5, 10.5]])),
              "(esperado [0 1])")
    mr = KNN(k=2, tarefa="regressao").fit(Xt, np.array([1.0, 3.0, 20.0, 40.0]))
    print("toy regressão:", mr.predict(np.array([[0.5, 0.5]])), "(esperado [2.])")


    X, y, _ = ler_arff("dados/kc1.arff", alvo_binario="true")
    tr, te = gerar_folds(y, 5, estratificado=True)[0]
    z = ZScore()
    Xtr, Xte = z.fit_transform(X[tr]), z.transform(X[te])
    for nome in DISTANCIAS:
        p = KNN(k=5, distancia=nome).fit(Xtr, y[tr]).predict(Xte)
        print(f"kc1 {nome}: acurácia fold 1 = {(p == y[te]).mean():.4f}")

    X, y, _ = ler_arff("dados/cpu_small.arff")
    tr, te = gerar_folds(y, 5)[0]
    z = ZScore()
    Xtr, Xte = z.fit_transform(X[tr]), z.transform(X[te])
    for nome in DISTANCIAS:
        p = KNN(k=5, distancia=nome, tarefa="regressao").fit(Xtr, y[tr]).predict(Xte)
        print(f"cpu_small {nome}: MAE fold 1 = {np.abs(p - y[te]).mean():.3f}")


if __name__ == "__main__":
    testar_knn()
