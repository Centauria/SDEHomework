# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial

import algo


def test_1():
    train_data = pd.read_csv("dataset.csv")

    x = train_data.math
    y = train_data.statistics

    print(algo.pearson(x, y))
    print(algo.spearman(x, y))


def check(x: pd.Series, y: pd.Series, sigma_range=(0, 100), p_range=(0, 0.5), points=200):
    plt.plot(x, y, 'x')
    plt.title('Original Data')
    plt.show()

    def r_with_gaussian_noise(method, sigma):
        return method(
            algo.add_gaussian_noise(x, sigma=sigma),
            algo.add_gaussian_noise(y, sigma=sigma)
        )

    def r_with_ps_noise(method, percentage):
        return method(
            algo.add_ps_noise(x, percentage, min_value=0, max_value=255),
            algo.add_ps_noise(y, percentage, min_value=0, max_value=255)
        )

    s = np.linspace(*sigma_range, points)
    p = np.linspace(*p_range, points)

    g_pearson = list(map(partial(r_with_gaussian_noise, algo.pearson), s))
    p_pearson = list(map(partial(r_with_ps_noise, algo.pearson), p))
    g_spearman = list(map(partial(r_with_gaussian_noise, algo.spearman), s))
    p_spearman = list(map(partial(r_with_ps_noise, algo.spearman), p))
    g_kendall = list(map(partial(r_with_gaussian_noise, algo.kendall), s))
    p_kendall = list(map(partial(r_with_ps_noise, algo.kendall), p))

    plt.plot(s, g_pearson, label='pearson', lw=0.5)
    plt.plot(s, g_spearman, label='spearman', lw=0.5)
    plt.plot(s, g_kendall, label='kendall', lw=0.5)
    plt.title('Gaussian Noise')
    plt.xlabel('sigma')
    plt.ylabel('r')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.show()
    plt.plot(p, p_pearson, label='pearson', lw=0.5)
    plt.plot(p, p_spearman, label='spearman', lw=0.5)
    plt.plot(p, p_kendall, label='kendall', lw=0.5)
    plt.title('Pulse Noise')
    plt.xlabel('percentage / %')
    plt.ylabel('r')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    x = pd.Series(np.arange(256))
    y = np.sqrt(x) * 16
    check(x, x)
    check(x, y)
    z = 2 * np.pi / 256 * x
    check(128 * (1 + np.cos(z)), 128 * (1 + np.cos(z + 0.1)))
    check(128 * (1 + np.cos(z)), 128 * (1 + np.cos(z + 1)))
