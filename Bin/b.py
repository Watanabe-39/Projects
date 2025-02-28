import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import sparse
import mglearn
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder

X, y = mglearn.datasets.make_wave(n_samples=100)
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)

# ビンを作る
bins = np.linspace(-3, 3, 11)   # -3から3までの間を10分割
print("bins: {}".format(bins))

# 個々のデータポイントがどのビンに入るかを記録
# digitize関数でビンに入るインデックスを取得
which_bin = np.digitize(X, bins=bins)
print("\nData points:\n", X[:5])
print("\nBin membership for data points:\n", which_bin[:5])

# scikit-learnモデルに適用するために
# which_bin(離散値特徴量)をワンホットエンコーディングに変換する
encoder = OneHotEncoder(sparse_output=False)
# encoder.fitでwhich_binに現れる整数値のバリエーションを確認
encoder.fit(which_bin)
# transformでワンホットエンコーディングを行う
X_binned = encoder.transform(which_bin)
print(X_binned[:5])

print("X_binned.shape: {}".format(X_binned.shape))

# 線形回帰モデルと決定木モデルをこのワンホットエンコーディングデータに対して作る
line_binned = encoder.transform(np.digitize(line, bins=bins))

reg = LinearRegression().fit(X_binned, y)
plt.plot(line, reg.predict(line_binned), label='linear regression binned')

reg = DecisionTreeRegressor(min_samples_split=3).fit(X_binned, y)
plt.plot(line, reg.predict(line_binned), label='decision tree binned')
plt.plot(X[:, 0], y, 'o', c='k')
plt.vlines(bins, -3, 3, linewidth=1, alpha=.2)
plt.legend(loc="best")
plt.ylabel("Regression output")
plt.xlabel("Input feature") 
plt.show()

# 結果: どちらも全く同じ
# 線形モデル -> 柔軟性が増加
# 決定木モデル -> 柔軟性が低下
# いくつかの特徴量が出力と非線形な関係を持つようなら、
# ビニングを使うとモデルの表現力を増強することができる。
# 線形モデルは、個々のビンに対して定数を学習する。