import numpy as np

from carregar_dados import ler_arff
from normalizacao import ZScore
from validacao_cruzada import gerar_folds
from knn import KNN, DISTANCIAS
from bayes_univariado import BayesUnivariado
from regressao_linear import RegressaoLinearMultipla


def saida_normalizacao():
    print("##### NORMALIZACAO #####")
    for nome, caminho, alvo in [
        ("kc1", "dados/kc1.arff", "true"),
        ("cpu_small", "dados/cpu_small.arff", None),
    ]:
        X, y, _ = ler_arff(caminho, alvo_binario=alvo)
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(X))
        corte = int(0.8 * len(X))
        X_tr, X_te = X[idx[:corte]], X[idx[corte:]]

        z = ZScore()
        X_tr_n = z.fit_transform(X_tr)
        X_te_n = z.transform(X_te)

        print(f"=== {nome} ===")
        print("Treino normalizado: media max |.|:", f"{np.abs(X_tr_n.mean(axis=0)).max():.2e}")
        print("Treino normalizado: desvios (min/max):",
              f"{X_tr_n.std(axis=0).min():.3f} / {X_tr_n.std(axis=0).max():.3f}")
        print("Teste normalizado: media max |.|:", f"{np.abs(X_te_n.mean(axis=0)).max():.3f}")
        print("Colunas constantes no treino:", int((X_tr.std(axis=0) == 0).sum()))


def saida_folds():
    print("\n##### FOLDS #####")
    X, y, _ = ler_arff("dados/kc1.arff", alvo_binario="true")
    print("=== kc1 (estratificado) ===")
    _imprimir_folds(y, gerar_folds(y, 5, estratificado=True))
    X, y, _ = ler_arff("dados/cpu_small.arff")
    print("=== cpu_small (simples) ===")
    _imprimir_folds(y, gerar_folds(y, 5, estratificado=False))


def _imprimir_folds(y, pares):
    n = len(y)
    todos_teste = np.concatenate([te for _, te in pares])
    print("Cada amostra esta em exatamente 1 teste:",
          len(todos_teste) == n and len(np.unique(todos_teste)) == n)
    for i, (tr, te) in enumerate(pares, 1):
        sem_sobreposicao = len(np.intersect1d(tr, te)) == 0
        print(f"Fold {i}: treino={len(tr)} teste={len(te)} "
              f"sem sobreposicao={sem_sobreposicao} media(y) teste={y[te].mean():.4f}")


def saida_knn():
    print("\n##### KNN #####")
    Xt = np.array([[0.0, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0]])
    yt = np.array([0, 0, 1, 1])
    for nome in DISTANCIAS:
        m = KNN(k=3, distancia=nome).fit(Xt, yt)
        print(f"toy {nome}:", m.predict(np.array([[0.5, 0.5], [10.5, 10.5]])), "(esperado [0 1])")
    mr = KNN(k=2, tarefa="regressao").fit(Xt, np.array([1.0, 3.0, 20.0, 40.0]))
    print("toy regressao:", mr.predict(np.array([[0.5, 0.5]])), "(esperado [2.])")

    X, y, _ = ler_arff("dados/kc1.arff", alvo_binario="true")
    tr, te = gerar_folds(y, 5, estratificado=True)[0]
    z = ZScore()
    Xtr, Xte = z.fit_transform(X[tr]), z.transform(X[te])
    for nome in DISTANCIAS:
        p = KNN(k=5, distancia=nome).fit(Xtr, y[tr]).predict(Xte)
        print(f"kc1 {nome}: acuracia fold 1 = {(p == y[te]).mean():.4f}")

    X, y, _ = ler_arff("dados/cpu_small.arff")
    tr, te = gerar_folds(y, 5)[0]
    z = ZScore()
    Xtr, Xte = z.fit_transform(X[tr]), z.transform(X[te])
    for nome in DISTANCIAS:
        p = KNN(k=5, distancia=nome, tarefa="regressao").fit(Xtr, y[tr]).predict(Xte)
        print(f"cpu_small {nome}: MAE fold 1 = {np.abs(p - y[te]).mean():.3f}")


def saida_bayes():
    print("\n##### BAYES UNIVARIADO #####")
    Xt = np.array([[1.0, 5.0], [1.2, 5.2], [0.8, 4.8], [5.0, 1.0], [5.2, 1.2], [4.8, 0.8]])
    yt = np.array([0, 0, 0, 1, 1, 1])
    m = BayesUnivariado().fit(Xt, yt)
    print("toy:", m.predict(np.array([[1.1, 5.1], [5.1, 1.1]])), "(esperado [0 1])")
    print("toy priori:", m.priori, "(esperado [0.5 0.5])")

    X, y, _ = ler_arff("dados/kc1.arff", alvo_binario="true")
    tr, te = gerar_folds(y, 5, estratificado=True)[0]
    m = BayesUnivariado().fit(X[tr], y[tr])
    p = m.predict(X[te])
    print(f"kc1: acuracia fold 1 = {(p == y[te]).mean():.4f}")
    print("kc1: priori (false/true) =", np.round(m.priori, 4))
    print("kc1: previsoes true no fold 1 =", int(p.sum()), "de", len(p))


def saida_regressao_linear():
    print("\n##### REGRESSAO LINEAR MULTIPLA #####")
    rng = np.random.default_rng(0)
    Xt = rng.normal(size=(50, 2))
    yt = 3.0 + 2.0 * Xt[:, 0] - 1.0 * Xt[:, 1]
    m = RegressaoLinearMultipla().fit(Xt, yt)
    print("toy coeficientes [b0 b1 b2]:", np.round(m.coef, 4), "(esperado [3. 2. -1.])")

    X, y, _ = ler_arff("dados/cpu_small.arff")
    tr, te = gerar_folds(y, 5)[0]
    z = ZScore()
    Xtr, Xte = z.fit_transform(X[tr]), z.transform(X[te])
    m = RegressaoLinearMultipla().fit(Xtr, y[tr])
    p = m.predict(Xte)
    ss_res = ((y[te] - p) ** 2).sum()
    ss_tot = ((y[te] - y[te].mean()) ** 2).sum()
    print(f"cpu_small: MAE fold 1 = {np.abs(p - y[te]).mean():.3f}")
    print(f"cpu_small: R2 fold 1 = {1 - ss_res / ss_tot:.4f}")


if __name__ == "__main__":
    saida_normalizacao()
    saida_folds()
    saida_knn()
    saida_bayes()
    saida_regressao_linear()
