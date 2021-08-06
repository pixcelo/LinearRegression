from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, Ridge, Lasso
#import pandas as pd

# 線形回帰（Ridge回帰、Lasso回帰）
bostn = load_boston()
X = bostn.data
y = bostn.target

# インスタンス
lr1 = LinearRegression()
lr1.fit(X,y)

# 線形回帰式の係数とその二乗和（cofe_は回帰変数）
print("LinearRegression")
for f, w in zip(bostn.feature_names, lr1.coef_):
    print("{0:7s}: {1:6.2f}".format(f, w))
print("coef = {0:4.2f}".format(sum(lr1.coef_**2)))

# Ridge回帰（インスタンス化の引数に「正規化の重みαを10」を設定）
lr2 = Ridge(alpha=10.0)
lr2.fit(X,y)
print("Ridge")
for f, w in zip(bostn.feature_names, lr2.coef_):
    print("{0:7s}: {1:6.2f}".format(f, w))
print("coef = {0:4.2f}".format(sum(lr2.coef_**2)))

# Losso回帰（インスタンス化の引数に「正規化の重みαを2.0」を設定）
# 係数が0になる変数が多くなる
lr3 = Lasso(alpha=2.0)
lr3.fit(X,y)
print("Lasso")
for f, w in zip(bostn.feature_names, lr3.coef_):
    print("{0:7s}: {1:6.2f}".format(f, w))
print("coef = {0:4.2f}".format(sum(lr3.coef_**2)))