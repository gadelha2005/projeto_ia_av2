import numpy as np
from collections import Counter
import re

def ler_arff(caminho, alvo_binario=None):
    atributos, X, y = [], [], []
    lendo_dados = False
    with open(caminho, encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha or linha.startswith("%"):
                continue
            if linha.lower().startswith("@attribute"):
                atributos.append(linha)
                continue
            if linha.lower().startswith("@data"):
                lendo_dados = True
                if alvo_binario is None and "{" in atributos[-1]:
                    raise ValueError(
                        "Alvo nominal em arquivo de regressão: "
                        f"{atributos[-1]}. Baixe a versão com alvo numérico."
                    )
                continue
            if not lendo_dados:
                continue
            partes = [p.strip() for p in linha.split(",")]
            X.append([float(v) for v in partes[:-1]])
            if alvo_binario:
                y.append(1 if partes[-1] == alvo_binario else 0)
            else:
                y.append(float(partes[-1]))
    return np.array(X), np.array(y), atributos


def checar_cpu_small(caminho="dados/cpu_small.arff"):
    print("=== cpu_small ===")
    tamanhos = Counter()
    lendo = False
    with open(caminho, encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha or linha.startswith("%"):
                continue
            if linha.lower().startswith("@data"):
                lendo = True
                continue
            if lendo:
                tamanhos[len(linha.split(","))] += 1
    print("Colunas por linha:", dict(tamanhos), "(esperado: uma só chave, 8192 linhas)")

    X, y, atributos = ler_arff(caminho)
    print("Atributos:", len(atributos))
    print("Último atributo:", atributos[-1], "(esperado: numeric)")
    print("X.shape:", X.shape, "(esperado: 8192 linhas; ~12 colunas, conferir)")
    print(f"alvo min/max/média: {y.min():.2f} / {y.max():.2f} / {y.mean():.2f}")
    print("Valores NaN em X:", int(np.isnan(X).sum()))
    print("Linhas duplicadas:", len(X) - len(np.unique(X, axis=0)))
    print("Valores distintos no alvo:", len(np.unique(y)),
          "(se for 2, é o alvo binarizado)")


def checar_kc1(caminho="dados/kc1.arff"):
    print("\n=== kc1 ===")
    X, y, atributos = ler_arff(caminho, alvo_binario="true")
    print("Último atributo:", atributos[-1])
    print("X.shape:", X.shape, "(esperado: (2109, 21))")
    print("y.sum():", int(y.sum()), "(esperado: 326)")
    print(f"Proporção de defeituosos: {y.mean():.4f} (esperado: ~0.1545)")
    print("Valores NaN em X:", int(np.isnan(X).sum()))
    print("Linhas duplicadas:", len(X) - len(np.unique(X, axis=0)))
    print("Primeira linha:", X[0][:4], "...")


if __name__ == "__main__":
    checar_cpu_small()
    checar_kc1()
    