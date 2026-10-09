import numpy as np


def matriz_confusao(y_real, y_prev):
    vp = int(((y_real == 1) & (y_prev == 1)).sum())
    vn = int(((y_real == 0) & (y_prev == 0)).sum())
    fp = int(((y_real == 0) & (y_prev == 1)).sum())
    fn = int(((y_real == 1) & (y_prev == 0)).sum())
    return vp, vn, fp, fn


def acuracia(y_real, y_prev):
    return float((y_real == y_prev).mean())


def precisao(y_real, y_prev):
    vp, vn, fp, fn = matriz_confusao(y_real, y_prev)
    return vp / (vp + fp) if vp + fp > 0 else 0.0


def recall(y_real, y_prev):
    vp, vn, fp, fn = matriz_confusao(y_real, y_prev)
    return vp / (vp + fn) if vp + fn > 0 else 0.0


def f1(y_real, y_prev):
    p = precisao(y_real, y_prev)
    r = recall(y_real, y_prev)
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def mae(y_real, y_prev):
    return float(np.abs(y_real - y_prev).mean())


def r2(y_real, y_prev):
    ss_res = ((y_real - y_prev) ** 2).sum()
    ss_tot = ((y_real - y_real.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot)


def r2_ajustado(y_real, y_prev, n_atributos):
    n = len(y_real)
    return float(1 - (1 - r2(y_real, y_prev)) * (n - 1) / (n - n_atributos - 1))
