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
