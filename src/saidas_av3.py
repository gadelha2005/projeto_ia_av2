import json

import numpy as np

from carregar_dados import ler_arff
from perceptron import PerceptronSimples
from mlp import MLPRegressor
from metricas import especificidade, mse, rmse
from experimentos import rodar_classificacao, rodar_regressao
from experimentos_av3 import (rodar_perceptron, rodar_mlp_topologias, rodar_mlp_hiperparametros,
                              rodar_impacto_zscore, TOPOLOGIAS_MLP)
from saidas import _formatar


def _tabela(resultados, colunas, titulo, extra=None):
    larg = max(len(n) for n in resultados) + 2
    cab = titulo.ljust(larg) + "".join(n.ljust(20) for n, _ in colunas)
    if extra:
        cab += extra[0]
    print(cab)
    for nome, m in resultados.items():
        linha = nome.ljust(larg) + "".join(_formatar(m[c]).ljust(20) for _, c in colunas)
        if extra:
            linha += str(extra[1](m))
        print(linha)


def saida_testes_unitarios():
    print("##### TESTES RAPIDOS #####")
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    p = PerceptronSimples(taxa_aprendizado=0.1, epocas=20).fit(X, np.array([0, 0, 0, 1]))
    print("perceptron AND:", p.predict(X), "(esperado [0 0 0 1])")
    print("perceptron AND erros/epoca:", p.erros_por_epoca[:8], "... (deve chegar a 0)")
    p = PerceptronSimples(taxa_aprendizado=0.1, epocas=20).fit(X, np.array([0, 1, 1, 0]))
    print("perceptron XOR erros/epoca:", p.erros_por_epoca[-5:], "(nunca zera: XOR nao e linearmente separavel)")

    rng = np.random.default_rng(0)
    Xs = rng.uniform(-2, 2, (500, 1))
    ys = 10 * np.sin(Xs[:, 0]) + 5
    m = MLPRegressor((20,), "tanh", 0.05, 300).fit(Xs, ys)
    print(f"mlp seno: MAE treino = {np.abs(m.predict(Xs) - ys).mean():.3f} (deve ser bem menor que 6)")

    real = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    prev = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0])
    print("especificidade:", round(especificidade(real, prev), 4), "(esperado 0.8333)")
    yr, yp = np.array([3.0, 5.0, 7.0, 9.0]), np.array([2.0, 5.0, 8.0, 9.0])
    print("mse:", mse(yr, yp), "(esperado 0.5)  rmse:", round(rmse(yr, yp), 4), "(esperado 0.7071)")


def _melhor(resultados, chave, maior=True):
    sinal = 1 if maior else -1
    return max(resultados, key=lambda n: sinal * resultados[n][chave][0])


def main():
    saida_testes_unitarios()
    tudo = {}

    X_cls, y_cls, _ = ler_arff("dados/kc1.arff", alvo_binario="true")
    X_reg, y_reg, _ = ler_arff("dados/cpu_small.arff")

    print("\n##### PERCEPTRON SIMPLES (kc1, 5-fold estratificado) #####")
    perc = rodar_perceptron(X_cls, y_cls)
    tudo["perceptron"] = perc
    _tabela(perc, [("Acuracia", "acuracia"), ("Recall", "recall"),
                   ("Especificidade", "especificidade"), ("Precisao", "precisao"),
                   ("F1-Score", "f1"), ("Acuracia treino", "acuracia_treino"),
                   ("Tempo Treino (s)", "tempo_treino")],
           "Configuracao", ("Matriz (VP VN FP FN)", lambda m: tuple(m["matriz"].values())))
    melhor_perc = _melhor(perc, "f1")
    print("Melhor por F1:", melhor_perc)

    print("\n##### MLP - TOPOLOGIAS (cpu_small, eta=0.01, 50 epocas) #####")
    topo = rodar_mlp_topologias(X_reg, y_reg)
    tudo["mlp_topologias"] = topo
    _tabela(topo, [("MSE", "mse"), ("RMSE", "rmse"), ("MAE", "mae"), ("R2", "r2"),
                   ("R2 ajustado", "r2_ajustado"), ("R2 treino", "r2_treino"),
                   ("Tempo Treino (s)", "tempo_treino"), ("Tempo Teste (s)", "tempo_teste")],
           "Topologia", ("Parametros", lambda m: m["n_parametros"]))
    melhor_topo = _melhor(topo, "rmse", maior=False)
    print("Melhor por RMSE:", melhor_topo)

    print(f"\n##### MLP - TAXA x EPOCAS na topologia {melhor_topo} #####")
    hip = rodar_mlp_hiperparametros(X_reg, y_reg, melhor_topo)
    tudo["mlp_hiperparametros"] = {"topologia": melhor_topo, "resultados": hip}
    _tabela(hip, [("RMSE", "rmse"), ("MAE", "mae"), ("R2", "r2"), ("R2 treino", "r2_treino"),
                  ("Tempo Treino (s)", "tempo_treino")], "Configuracao")
    melhor_hip = _melhor(hip, "rmse", maior=False)
    print("Melhor por RMSE:", melhor_hip)

    print("\n##### COMPARACAO COM A AV2 (5-fold, com Z-score) #####")
    av2_cls = rodar_classificacao(X_cls, y_cls)
    av2_cls[f"Perceptron ({melhor_perc})"] = perc[melhor_perc]
    _tabela(av2_cls, [("Acuracia", "acuracia"), ("Precisao", "precisao"), ("Recall", "recall"),
                      ("F1-Score", "f1"), ("Tempo Treino (s)", "tempo_treino"),
                      ("Tempo Teste (s)", "tempo_teste")], "Classificador")
    av2_reg = rodar_regressao(X_reg, y_reg)
    av2_reg[f"MLP {melhor_topo} ({melhor_hip})"] = hip[melhor_hip]
    print()
    _tabela(av2_reg, [("R2", "r2"), ("R2 ajustado", "r2_ajustado"), ("MAE", "mae"),
                      ("Tempo Treino (s)", "tempo_treino"), ("Tempo Teste (s)", "tempo_teste")],
           "Regressor")
    tudo["comparacao"] = {"classificacao": av2_cls, "regressao": av2_reg}

    print("\n##### IMPACTO DO Z-SCORE (com x sem) #####")
    taxa_p, epocas_p = [float(v.split("=")[1]) for v in melhor_perc.split(" | ")]
    taxa_m, epocas_m = [float(v.split("=")[1]) for v in melhor_hip.split(" | ")]
    z = rodar_impacto_zscore(
        X_cls, y_cls, X_reg, y_reg,
        dict(taxa_aprendizado=taxa_p, epocas=int(epocas_p)),
        dict(**TOPOLOGIAS_MLP[melhor_topo], taxa_aprendizado=taxa_m, epocas=int(epocas_m)))
    tudo["zscore"] = z
    print("Classificacao (kc1): F1 e acuracia")
    for nome, r in z["classificacao"].items():
        print(f"  {nome.ljust(32)} com: F1 {_formatar(r['com_zscore']['f1'])}  acc {_formatar(r['com_zscore']['acuracia'])}"
              f" | sem: F1 {_formatar(r['sem_zscore']['f1'])}  acc {_formatar(r['sem_zscore']['acuracia'])}")
    print("Regressao (cpu_small): RMSE e R2")
    for nome, r in z["regressao"].items():
        print(f"  {nome.ljust(32)} com: RMSE {_formatar(r['com_zscore']['rmse'])}  R2 {_formatar(r['com_zscore']['r2'])}"
              f" | sem: RMSE {_formatar(r['sem_zscore']['rmse'])}  R2 {_formatar(r['sem_zscore']['r2'])}")

    with open("resultados_av3.json", "w", encoding="utf-8") as f:
        json.dump(tudo, f, indent=1, ensure_ascii=False, default=float)
    print("\nResultados salvos em resultados_av3.json")


if __name__ == "__main__":
    main()
