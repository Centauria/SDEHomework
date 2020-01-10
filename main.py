import pandas as pd

from algo import pearson, spearman

train_data = pd.read_csv("dataset.csv")

x = train_data.math
y = train_data.statistics

print(pearson(x, y))
print(spearman(x, y))
