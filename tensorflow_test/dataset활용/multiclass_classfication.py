#데이터labels 총 46개
#데이터들은 훈련샘플  8982, 테스트샘플 2246

import tensorflow as tf
from tensorflow.keras.datasets import reuters
import numpy as np
from tensorflow.keras import models
from tensorflow.keras import layers
import matplotlib.pyplot as plt


(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words = 10000)
print(len(train_data), len(test_data))

word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_newswire = ''.join([reverse_word_index.get(i-3,'?') for i in train_data[0]])

def vectorize_sequences(sequences, dimension = 10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i,sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

one_hot_train_labels = tf.keras.utils.to_categorical(train_labels)
one_hot_test_labels = tf.keras.utils.to_categorical(test_labels)
# 모델생성
model = models.Sequential()
model.add(layers.Dense(64, activation = 'relu', input_shape = (10000,)))
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(46, activation ='softmax')) # OUTPUT클래스 개수


model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
x_val=x_train[:1000]
partial_x_train= x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

history = model.fit(partial_x_train, partial_y_train,epochs = 9, batch_size = 512, validation_data= (x_val,y_val))
results = model.evaluate(x_test,one_hot_test_labels)

print(results)

predictions = model.predict(x_test)
print(np.argmax(predictions[990])) # 각 인덱스마다 예측하는 값이 다르다.
