import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import sparse
import mglearn
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.svm import SVR

# カーネル法を使ったSVMモデルを、変換していないオリジナルデータに適用してみる

X, y = mglearn.datasets.make_wave(n_samples=100)
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)

for gamma in [1, 10]:
    svr = SVR(gamma=gamma).fit(X, y)
    plt.plot(line, svr.predict(line), label='SVR gamma={}'.format(gamma))
plt.plot(X[:, 0], y, 'o', c='k')
plt.ylabel("Regression output")
plt.xlabel("Input feature")
plt.legend(loc="best")
plt.show()

# より複雑なモデルを用いると、特徴量に対して明示的な変換を行わなくても、
# 多項式回帰と同じように複雑な予測をすることができる