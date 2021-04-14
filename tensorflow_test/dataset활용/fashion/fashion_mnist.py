import tensorflow as tf
import matplotlib.pyplot as plt

# 1. Fashion MNIST 데이터셋 불러오기
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_X, train_Y) ,(test_X, test_Y) = fashion_mnist.load_data()
print(len(train_X),len(train_Y))

plt.imshow(train_X[3], cmap='gray') # 패션 이미지 출력
plt.colorbar()
plt.show()
print('분류 번호 : ',train_Y[3])


#rbg 색상정규화하기
train_X = train_X / 255.0
test_X = test_X / 255.0

print(train_X[3])

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(units = 128, activation = 'relu'),
    tf.keras.layers.Dense(units = 10, activation = 'softmax')
])

model.compile(optimizer = tf.keras.optimizers.Adam(),
             loss='sparse_categorical_crossentropy', # sparse -> 희소행렬없이 바로 정답행렬로 사용가능
              metrics = ['accuracy'])

model.summary()
history = model.fit(train_X, train_Y, epochs = 25, validation_split= 0.25)

#그래프
plt.figure(figsize = (12,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], 'b-', label = 'loss')
plt.plot(history.history['val_loss'],'r--', label = 'val_loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], 'g--', label='accuracy')
plt.plot(history.history['val_accuracy'], 'k--', label='val_accuracy')
plt.xlabel('Epoch')
plt.ylim(0.7,1)
plt.legend()

plt.show()

result = model.evaluate(test_X, test_Y)
print('정확도', result[1]* 100 ,'%')

