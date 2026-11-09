import time

import numpy as np

from normalizacao import ZScore
from validacao_cruzada import gerar_folds
from experimentos import validar, MODELOS_CLASSIFICACAO, MODELOS_REGRESSAO
from perceptron import PerceptronSimples
from mlp import MLPRegressor
from metricas import (matriz_confusao, acuracia, precisao, recall, especificidade, f1,
                      mse, rmse, mae, r2, r2_ajustado)

METRICAS_CLASSIFICACAO = {
    "acuracia": acuracia,
    "precisao": precisao,
    "recall": recall,
    "especificidade": especificidade,
    "f1": f1,
}

TAXAS_PERCEPTRON = [0.001, 0.01, 0.1, 1.0]
EPOCAS_PERCEPTRON = [10, 50, 100]

TOPOLOGIAS_MLP = {
    "A: [5] tanh": dict(camadas_ocultas=(5,), ativacao="tanh"),
    "B: [20] tanh": dict(camadas_ocultas=(20,), ativacao="tanh"),
    "C: [50] tanh": dict(camadas_ocultas=(50,), ativacao="tanh"),
    "D: [20, 10] tanh": dict(camadas_ocultas=(20, 10), ativacao="tanh"),
    "E: [20, 10] relu": dict(camadas_ocultas=(20, 10), ativacao="relu"),
    "F: [20, 10] sigmoide": dict(camadas_ocultas=(20, 10), ativacao="sigmoide"),
    "G: [32, 16, 8] relu": dict(camadas_ocultas=(32, 16, 8), ativacao="relu"),
}
TAXAS_MLP = [0.001, 0.01, 0.1]
EPOCAS_MLP = [10, 50, 100]
TAXA_MLP_BASE = 0.01
EPOCAS_MLP_BASE = 50


def metricas_regressao(n_atributos):
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "r2_ajustado": lambda real, prev: r2_ajustado(real, prev, n_atributos),
    }


def matriz_confusao_total(criar_modelo, X, y, folds, normalizar=True):
    total = np.zeros(4, dtype=int)
    for idx_tr, idx_te in folds:
        X_tr, X_te = X[idx_tr], X[idx_te]
        if normalizar:
            z = ZScore()
            X_tr, X_te = z.fit_transform(X_tr), z.transform(X_te)
        y_prev = criar_modelo().fit(X_tr, y[idx_tr]).predict(X_te)
        total += np.array(matriz_confusao(y[idx_te], y_prev))
    vp, vn, fp, fn = total
    return {"vp": int(vp), "vn": int(vn), "fp": int(fp), "fn": int(fn)}


def rodar_perceptron(X, y, k_folds=5):
    folds = gerar_folds(y, k_folds, estratificado=True)
    resultados = {}
    for taxa in TAXAS_PERCEPTRON:
        for epocas in EPOCAS_PERCEPTRON:
            criar = lambda t=taxa, e=epocas: PerceptronSimples(taxa_aprendizado=t, epocas=e)
            nome = f"eta={taxa} | epocas={epocas}"
            res = validar(criar, X, y, folds, METRICAS_CLASSIFICACAO, avaliar_treino=True)
            res["matriz"] = matriz_confusao_total(criar, X, y, folds)
            resultados[nome] = res
    return resultados


def rodar_mlp_topologias(X, y, k_folds=5):
    folds = gerar_folds(y, k_folds, estratificado=False)
    metricas = metricas_regressao(X.shape[1])
    resultados = {}
    for nome, cfg in TOPOLOGIAS_MLP.items():
        criar = lambda c=cfg: MLPRegressor(**c, taxa_aprendizado=TAXA_MLP_BASE, epocas=EPOCAS_MLP_BASE)
        res = validar(criar, X, y, folds, metricas, avaliar_treino=True)
        res["n_parametros"] = contar_parametros(X.shape[1], cfg["camadas_ocultas"])
        resultados[nome] = res
    return resultados


def rodar_mlp_hiperparametros(X, y, topologia, k_folds=5):
    folds = gerar_folds(y, k_folds, estratificado=False)
    metricas = metricas_regressao(X.shape[1])
    cfg = TOPOLOGIAS_MLP[topologia]
    resultados = {}
    for taxa in TAXAS_MLP:
        for epocas in EPOCAS_MLP:
            criar = lambda t=taxa, e=epocas: MLPRegressor(**cfg, taxa_aprendizado=t, epocas=e)
            resultados[f"eta={taxa} | epocas={epocas}"] = validar(
                criar, X, y, folds, metricas, avaliar_treino=True)
    return resultados


def contar_parametros(n_entradas, camadas_ocultas):
    tamanhos = [n_entradas, *camadas_ocultas, 1]
    return int(sum(a * b + b for a, b in zip(tamanhos[:-1], tamanhos[1:])))


def rodar_impacto_zscore(X_cls, y_cls, X_reg, y_reg, cfg_perceptron, cfg_mlp, k_folds=5):
    folds_cls = gerar_folds(y_cls, k_folds, estratificado=True)
    folds_reg = gerar_folds(y_reg, k_folds, estratificado=False)
    modelos_cls = dict(MODELOS_CLASSIFICACAO)
    modelos_cls["Perceptron Simples"] = lambda: PerceptronSimples(**cfg_perceptron)
    modelos_reg = dict(MODELOS_REGRESSAO)
    modelos_reg["MLP"] = lambda: MLPRegressor(**cfg_mlp)
    metricas_reg = metricas_regressao(X_reg.shape[1])

    resultados = {"classificacao": {}, "regressao": {}}
    with np.errstate(all="ignore"):
        for nome, criar in modelos_cls.items():
            resultados["classificacao"][nome] = {
                "com_zscore": validar(criar, X_cls, y_cls, folds_cls, METRICAS_CLASSIFICACAO, True),
                "sem_zscore": validar(criar, X_cls, y_cls, folds_cls, METRICAS_CLASSIFICACAO, False),
            }
        for nome, criar in modelos_reg.items():
            resultados["regressao"][nome] = {
                "com_zscore": validar(criar, X_reg, y_reg, folds_reg, metricas_reg, True),
                "sem_zscore": validar(criar, X_reg, y_reg, folds_reg, metricas_reg, False),
            }
    return resultados
