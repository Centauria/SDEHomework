import pandas as pd
import numpy as np

train_data = pd.read_csv("dataset.csv", encoding='ANSI')

x = train_data['math']
y = train_data['statistics']

L = len(x)

# 计算spearman相关系数
sort = x.sort_values(ascending=False)
x = pd.DataFrame(sort)
x['rank'] = [i + 1 for i in range(L)]  # 生成秩
x = x.sort_index()  # 合成一个dataframe

sort = y.sort_values(ascending=False)
y = pd.DataFrame(sort)
y['rank'] = [i + 1 for i in range(L)]  # 生成秩
y = y.sort_index()  # 合成一个dataframe

d_2 = []
for i in range(L):
    d_2.append((x['rank'][i] - y['rank'][i]) ** 2)

R = 3 * np.sum(d_2) / (L * (L ** 2 - 1))  # 等级相关系数？？？
Rs = 1 - 2 * R  # 相关等级系数？？最后结果
