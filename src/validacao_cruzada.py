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


def testar_folds():
    from carregar_dados import ler_arff

    X, y, _ = ler_arff("dados/kc1.arff", alvo_binario="true")
    print("=== kc1 (estratificado) ===")
    _checar(y, gerar_folds(y, 5, estratificado=True))

    X, y, _ = ler_arff("dados/cpu_small.arff")
    print("=== cpu_small (simples) ===")
    _checar(y, gerar_folds(y, 5, estratificado=False))


def _checar(y, pares):
    n = len(y)
    todos_teste = np.concatenate([te for _, te in pares])
    print("Cada amostra está em exatamente 1 teste:",
          len(todos_teste) == n and len(np.unique(todos_teste)) == n)
    for i, (tr, te) in enumerate(pares, 1):
        sem_sobreposicao = len(np.intersect1d(tr, te)) == 0
        print(f"Fold {i}: treino={len(tr)} teste={len(te)} "
              f"sem sobreposição={sem_sobreposicao} "
              f"média(y) teste={y[te].mean():.4f}")


if __name__ == "__main__":
    testar_folds()
