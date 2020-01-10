# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def pearson(x: pd.Series, y: pd.Series):
    assert len(x) == len(y)
    m_x, m_y = x.mean(), y.mean()
    x_, y_ = x - m_x, y - m_y
    xy = x_ * y_
    xx = x_ ** 2
    yy = y_ ** 2
    r = xy.sum() / np.sqrt(xx.sum() * yy.sum())
    return r


def spearman(x: pd.Series, y: pd.Series):
    assert len(x) == len(y)
    L = len(x)

    def ranks(z: pd.Series) -> pd.Series:
        s = z.sort_values(ascending=False)
        zs = pd.DataFrame(s)
        zs['r'] = [i + 1 for i in range(len(z))]
        zs = zs.sort_index()
        return zs['r']

    d2 = (ranks(x) - ranks(y)) ** 2
    r = 3 * d2.sum() / L / (L * L - 1)
    rs = 1 - 2 * r
    return rs


def kendall(x: pd.Series, y: pd.Series):
    assert len(x) == len(y)
    L = len(x)
    x_, y_ = x.to_numpy().reshape(1, -1), y.to_numpy().reshape(1, -1)
    xd = x_ - x_.transpose()
    yd = y_ - y_.transpose()
    N = np.sum(np.triu(np.sign(xd) * np.sign(yd)))
    tau = 2 * N / (L * (L - 1))
    return tau


def add_gaussian_noise(x: pd.Series, miu=0.0, sigma=1.0):
    noise = np.random.randn(len(x)) * sigma + miu
    return x + noise


def add_ps_noise(x: pd.Series, percentage: float):
    assert 0 <= percentage <= 1
    positions = np.random.rand(len(x)) < percentage
    maximum, minimum = x.max(), x.min()
    result = x.copy()
    result[positions] = np.random.choice((maximum, minimum), np.sum(positions))
    return result
