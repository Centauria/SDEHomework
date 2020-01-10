# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import algo


def test_1():
    train_data = pd.read_csv("dataset.csv")

    x = train_data.math
    y = train_data.statistics

    print(algo.pearson(x, y))
    print(algo.spearman(x, y))


def test_2():
    x = pd.Series(np.arange(1, 101))
    y = np.sqrt(x) * 10

    def spearman_with_gaussian_noise(sigma):
        return algo.spearman(
            algo.add_gaussian_noise(x, sigma=sigma),
            algo.add_gaussian_noise(y, sigma=sigma)
        )

    def spearman_with_ps_noise(percentage):
        return algo.spearman(
            algo.add_ps_noise(x, percentage),
            algo.add_ps_noise(y, percentage)
        )

    s = np.arange(0, 20, 0.1)
    p = np.arange(0, 0.5, 0.01)

    g_list = list(map(spearman_with_gaussian_noise, s))
    p_list = list(map(spearman_with_ps_noise, p))

    plt.plot(s, g_list)
    plt.show()
    plt.plot(p, p_list)
    plt.show()


if __name__ == '__main__':
    test_2()
