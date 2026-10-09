import numpy as np


def gerar_folds(y, k=5, estratificado=False, semente=42):
    rng = np.random.default_rng(semente)
    n = len(y)
    folds = [[] for _ in range(k)]

    if estratificado:
        for classe in np.unique(y):
            idx = rng.permutation(np.where(y == classe)[0])
            for i, j in enumerate(idx):
                folds[i % k].append(j)
    else:
        idx = rng.permutation(n)
        for i, j in enumerate(idx):
            folds[i % k].append(j)

    folds = [np.array(sorted(f)) for f in folds]
    pares = []
    for i in range(k):
        treino = np.concatenate([folds[j] for j in range(k) if j != i])
        pares.append((np.sort(treino), folds[i]))
    return pares
