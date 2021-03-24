import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#fixed acidity -> 주석산, volatile acidity ->초산, citric acid ->구연산, residual sugar ->당도, chlorides->염화물(소금)
red = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';')
white = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv', sep=';')

red['type'] = 0
white['type'] = 1

wine = pd.concat([red,white]) # 데이터프레임 합치기


#데이터 정규화하기
wine_norm = (wine - wine.min()) / (wine.max() - wine.min())
#print(wine_norm.head())
#print(wine_norm.describe())

wine_shuffle = wine_norm.sample(frac = 1) # frac만큼 랜덤으로 비율정해준다
print(wine_shuffle.head())
wine_np = wine_shuffle.to_numpy() #numpy로 변환하기
print(wine_np[:5])

#학습 train,test만들기
train_idx = int(len(wine_np) * 0.8)#test 비율 8:2로 맞추기
train_X, train_Y = wine_np[:train_idx, :-1], wine_np[:train_idx, -1]
test_X, test_Y =  wine_np[train_idx:, :-1], wine_np[train_idx:,-1]
print(train_X[0])
print(train_Y[0])
print(test_X[0])
print(test_Y[0])

#분류할때 사용하기, 원핫인코딩사용
train_Y = tf.keras.utils.to_categorical(train_Y, num_classes= 2)
test_Y = tf.keras.utils.to_categorical(test_Y, num_classes= 2)
print(train_Y[0])
print(test_Y[0])

#모델 학습 만들기
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units = 48, activation = 'relu', input_shape =(12,)),
    tf.keras.layers.Dense(units = 24, activation = 'relu'),
    tf.keras.layers.Dense(units = 12, activation = 'relu'),
    tf.keras.layers.Dense(units = 2, activation = 'softmax')
])

model.compile(optimizer = tf.keras.optimizers.Adam(lr = 0.07), loss='categorical_crossentropy', metrics = ['accuracy'])

model.summary()

history = model.fit(train_X, train_Y, epochs = 25, batch_size = 32, validation_split = 0.25)

#그래프 시각화
plt.figure(figsize = (12,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], 'b-', label ='loss')
plt.plot(history.history['val_loss'],'r--', label='val_loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'],'g-',label='accuracy')
plt.plot(history.history['val_accuracy'], 'k--',label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylim(0.7,1)
plt.legend()

plt.show()

#평가하기
model.evaluate(test_X,test_Y) #정확도보기

