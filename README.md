# Projeto IA AV2 + AV3

Implementações manuais (só NumPy, sem scikit-learn/pandas) de algoritmos de classificação e regressão.

- Classificação: `dados/kc1.arff` (2109 x 21, alvo `defects`)
- Regressão: `dados/cpu_small.arff` (8192 x 12, alvo `usr`)
- Validação cruzada 5-fold (estratificada na classificação), Z-score ajustado só no treino de cada fold.

## AV2
kNN (Euclidiana/Manhattan), Bayes univariado, regressão kNN e regressão linear múltipla.

```bash
python src/saidas.py
```

## AV3
- `src/perceptron.py`: Perceptron Simples (degrau, regra delta), grade de taxa de aprendizado x épocas.
- `src/mlp.py`: MLP para regressão com backpropagation e mini-lotes; camadas ocultas, neurônios, ativação (tanh, relu, sigmoide), taxa e épocas configuráveis.
- `src/metricas.py`: + especificidade, MSE e RMSE.
- `src/experimentos_av3.py`: experimentos do Perceptron, topologias do MLP, taxa x épocas e impacto do Z-score (com x sem).

```bash
python src/saidas_av3.py   # ~2 min
```
