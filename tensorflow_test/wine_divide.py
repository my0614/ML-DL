import pandas as pd
#fixed acidity -> 주석산, volatile acidity ->초산, citric acid ->구연산, residual sugar ->당도, chlorides->염화물(소금)
red = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';')
white = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv', sep=';')

red['type'] = 0
white['type'] = 1

wine = pd.concat([red,white]) # 데이터프레임 합치기
pd.DataFrame(wine.describe())

import matplotlib.pyplot as plt
plt.hist(wine['type'])
plt.xticks([0,1])
plt.show()

print(wine['type'].value_counts())
