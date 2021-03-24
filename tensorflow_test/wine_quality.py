import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#fixed acidity -> 주석산, volatile acidity ->초산, citric acid ->구연산, residual sugar ->당도, chlorides->염화물(소금)
red = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';')
white = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv', sep=';')

red['type'] = 0
white['type'] = 0
wine = pd.concat([red,white]) #데이터합치기

#와인품질에 대한 정보
print(wine['quality'].describe()) 
print(wine['quality'].value_counts)

plt.hist(wine['quality'], bins = 7, rwidth = 0.5)
plt.xlabel('quality')
plt.show()

#새로운 품질 단계만들기
wine.loc[wine['quality'] <= 5, 'new_quality'] = 0 
wine.loc[wine['quality'] == 6 , 'new_quality'] = 1 
wine.loc[wine['quality'] >= 7, 'new_quality'] = 2 

print(wine['new_quality'].describe()) 
print(wine['new_quality'].value_counts())

del wine['quality']
wine_norm = (wine-wine.min()) / (wine.max() - wine.min())
print('wine_norm',wine_norm)

wine_shuffle = wine_norm.sample(frac = 1)
wine_np = wine_shuffle.to_numpy()

train_idx = int(len(wine_np) * 0.8)
train_X, train_Y = wine_np[:train_idx, :-1], wine_np[:train_idx, -1]
test_X, test_Y = wine_np[train_idx : , :-1], wine_np[train_idx:, -1]
train_Y = tf.keras.utils.to_categorical(train_Y, num_classes = 3) # num_classes = 3인 이유는 품질단계 0,1,2
test_Y = tf.keras.utils.to_categorical(test_Y, num_classes = 3)

#모델학습
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units = 48, activation = 'relu', input_shape = (12,)),
    tf.keras.layers.Dense(units = 24, activation = 'relu'),
    tf.keras.layers.Dense(units = 12, activation = 'relu'),
    tf.keras.layers.Dense(units = 3, activation = 'softmax')
])

model.compile(optimizer = tf.keras.optimizers.Adam(lr= 0.07), loss= 'categorical_crossentropy', metrics = 'accuray') # 분류의 metrics ->정확도
