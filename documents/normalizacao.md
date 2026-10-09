
# Normalização (z-score): código + teoria
 
Cada trecho do código (C1, C2...) aponta para a ideia teórica (T1, T2...) que o justifica.
 
## Código (`src/normalizacao.py`)
 
**C1. A classe `ZScore` tem dois passos: `fit` e `transform`.**
```python
z = ZScore()
X_tr_n = z.fit_transform(X_tr)   # aprende média/desvio NO TREINO e aplica
X_te_n = z.transform(X_te)       # só aplica, com os valores do treino
```
→ teoria T3 (vazamento de dados).
 
**C2. `fit` calcula média e desvio por coluna.**
```python
self.media  = X.mean(axis=0)
self.desvio = X.std(axis=0)
```
`axis=0` = uma média/desvio para cada atributo (coluna). → teoria T2.
 
**C3. Proteção contra coluna constante.**
```python
self.desvio[self.desvio == 0] = 1.0
```
Se todos os valores de uma coluna são iguais, o desvio é 0 e daria divisão por zero. Com 1.0 a coluna vira só zeros. → teoria T4.
 
**C4. `transform` aplica a fórmula.**
```python
return (X - self.media) / self.desvio
```
Numpy faz broadcasting: subtrai/divide cada coluna pelo seu próprio valor. → teoria T2.
 
**C5. `testar_normalizacao()`** simula um fold 80/20 em kc1 e cpu_small e confere: média ≈ 0 e desvio = 1 no treino. No teste a média NÃO é exatamente 0 (e isso é o esperado, ver T3).
Rodar da raiz do repo: `python3 -c "import sys; sys.path.insert(0,'src'); from normalizacao import testar_normalizacao as t; t()"`
 
## Teoria
 
**T1. Por que normalizar?** kNN usa distância. Se um atributo vai de 0 a 2.000.000 (ex.: `freeswap` no cpu_small) e outro de 0 a 5 (`fork`), a distância é dominada só pelo primeiro e os demais viram ruído. Normalizar coloca todos na mesma escala. (Regressão linear não exige, mas não prejudica e deixa os coeficientes comparáveis. Bayes gaussiano também não exige, pois estima média/desvio por atributo.)
 
**T2. Fórmula z-score.** `z = (x − μ) / σ`. Resultado: média 0 e desvio padrão 1 em cada atributo. Mede "quantos desvios padrão o valor está da média". Mantém a forma da distribuição (diferente do min-max, que é sensível a outliers).
 
**T3. Vazamento de dados (data leakage).** Dentro da validação cruzada, μ e σ devem ser calculados só com os dados de treino do fold e reaproveitados no teste. Se usarmos o dataset inteiro, o teste "espiou" informação do treino e a métrica fica otimista. Por isso o teste normalizado não tem média exatamente 0: é o comportamento correto.
 
**T4. Desvio zero.** Se σ = 0 a fórmula é indefinida. Como a coluna não carrega informação (é constante), basta zerá-la. Nos nossos dois datasets não há colunas constantes (o teste imprime 0).
 
## Como isso entra nos próximos passos
No loop de 5-fold: para cada fold, `z.fit_transform(X_treino)` e `z.transform(X_teste)`, e só depois treinar/avaliar kNN. O alvo `y` não é normalizado (na regressão, o R² e o MAE ficam na escala original).
 
