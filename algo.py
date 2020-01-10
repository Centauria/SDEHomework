# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def pearson(x: pd.Series, y: pd.Series):
    m_x, m_y = x.mean(), y.mean()
    x_, y_ = x - m_x, y - m_y
    xy = x_ * y_
    xx = x_ ** 2
    yy = y_ ** 2
    r = xy.sum() / np.sqrt(xx.sum() * yy.sum())
    return r
