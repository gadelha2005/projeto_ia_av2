import time

import numpy as np

from normalizacao import ZScore
from validacao_cruzada import gerar_folds
from knn import KNN
from bayes_univariado import BayesUnivariado
from regressao_linear import RegressaoLinearMultipla
from metricas import acuracia, precisao, recall, f1, mae, r2, r2_ajustado

MODELOS_CLASSIFICACAO = {
    "K-vizinhos (Dist. Euclidiana)": lambda: KNN(k=5, distancia="euclidiana"),
    "K-vizinhos (Dist. Manhattan)": lambda: KNN(k=5, distancia="manhattan"),
    "Bayesiano (Univariado)": lambda: BayesUnivariado(),
}

MODELOS_REGRESSAO = {
    "K-vizinhos (Dist. Euclidiana)": lambda: KNN(k=5, distancia="euclidiana", tarefa="regressao"),
    "K-vizinhos (Dist. Manhattan)": lambda: KNN(k=5, distancia="manhattan", tarefa="regressao"),
    "Regressao Linear Multipla": lambda: RegressaoLinearMultipla(),
}


def validar(criar_modelo, X, y, folds, metricas):
    valores = {nome: [] for nome in metricas}
    valores["tempo_treino"] = []
    valores["tempo_teste"] = []

    for idx_tr, idx_te in folds:
        z = ZScore()
        X_tr = z.fit_transform(X[idx_tr])
        X_te = z.transform(X[idx_te])

        modelo = criar_modelo()
        t0 = time.perf_counter()
        modelo.fit(X_tr, y[idx_tr])
        t1 = time.perf_counter()
        y_prev = modelo.predict(X_te)
        t2 = time.perf_counter()

        for nome, funcao in metricas.items():
            valores[nome].append(funcao(y[idx_te], y_prev))
        valores["tempo_treino"].append(t1 - t0)
        valores["tempo_teste"].append(t2 - t1)

    return {nome: (float(np.mean(v)), float(np.std(v))) for nome, v in valores.items()}


def rodar_classificacao(X, y, k_folds=5):
    folds = gerar_folds(y, k_folds, estratificado=True)
    metricas = {"acuracia": acuracia, "precisao": precisao, "recall": recall, "f1": f1}
    return {nome: validar(criar, X, y, folds, metricas)
            for nome, criar in MODELOS_CLASSIFICACAO.items()}


def rodar_regressao(X, y, k_folds=5):
    folds = gerar_folds(y, k_folds, estratificado=False)
    metricas = {
        "r2": r2,
        "r2_ajustado": lambda real, prev: r2_ajustado(real, prev, X.shape[1]),
        "mae": mae,
    }
    return {nome: validar(criar, X, y, folds, metricas)
            for nome, criar in MODELOS_REGRESSAO.items()}
