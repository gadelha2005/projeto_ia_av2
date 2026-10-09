import numpy as np


class ZScore:
    """Padronização z-score: z = (x - media) / desvio.

    Os parâmetros (média e desvio) são aprendidos SOMENTE no treino (fit)
    e depois aplicados a qualquer conjunto (transform), evitando vazamento
    de dados do teste para o treino.
    """

    def __init__(self):
        self.media = None
        self.desvio = None

    def fit(self, X):
        self.media = X.mean(axis=0)
        self.desvio = X.std(axis=0)
        # Coluna constante teria desvio 0 (divisão por zero): usa 1,
        # o que deixa a coluna toda zerada após subtrair a média.
        self.desvio[self.desvio == 0] = 1.0
        return self

    def transform(self, X):
        if self.media is None:
            raise RuntimeError("Chame fit() antes de transform().")
        return (X - self.media) / self.desvio

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def testar_normalizacao():
    from carregar_dados import ler_arff

    for nome, caminho, alvo in [
        ("kc1", "dados/kc1.arff", "true"),
        ("cpu_small", "dados/cpu_small.arff", None),
    ]:
        X, y, _ = ler_arff(caminho, alvo_binario=alvo)

        # Simula um fold: 80% treino, 20% teste
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(X))
        corte = int(0.8 * len(X))
        X_tr, X_te = X[idx[:corte]], X[idx[corte:]]

        z = ZScore()
        X_tr_n = z.fit_transform(X_tr)
        X_te_n = z.transform(X_te)

        print(f"=== {nome} ===")
        print("Treino normalizado: média máx |.|:",
              f"{np.abs(X_tr_n.mean(axis=0)).max():.2e}")
        print("Treino normalizado: desvios (min/max):",
              f"{X_tr_n.std(axis=0).min():.3f} / {X_tr_n.std(axis=0).max():.3f}")
        print("Teste normalizado: média máx |.|:",
              f"{np.abs(X_te_n.mean(axis=0)).max():.3f}",
              "(não precisa ser 0: usa parâmetros do treino)")
        print("Colunas constantes no treino:", int((X_tr.std(axis=0) == 0).sum()))


if __name__ == "__main__":
    testar_normalizacao()
