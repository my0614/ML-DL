import tensorflow as tf
import matplotlib.pyplot as plt


fashion_mnist = tf.keras.datasets.fashion_mnist
(train_X,train_Y) , (test_X,test_Y) = fashion_mnist.load_data()

train_X = train_X / 255.0
test_X = test_X / 255.0

print(train_X.shape, test_X.shape)
train_X = train_X.reshape(-1,28,28,1)
test_X = test_X.reshape(-1,28,28,1)

#데이터크기
print(train_X.shape, test_X.shape)

plt.figure(figsize = (10,10))

#16개의 이미지 출력하기
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.imshow(train_X[i].reshape(28,28), cmap = 'gray')
plt.show()
print(train_Y[:16])

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(input_shape=(28,28,1), kernel_size = (3,3), filters = 16),
    tf.keras.layers.MaxPool2D(strides = (2,2)),
    tf.keras.layers.Conv2D(kernel_size=(3,3), filters = 32),
    tf.keras.layers.MaxPool2D(strides = (2,2)),
    tf.keras.layers.Conv2D(kernel_size = (3,3), filters = 64),
    tf.keras.layers.Flatten(), # 이미지이므로 차원 줄이기
    tf.keras.layers.Dense(units=  128, activation = 'relu'),
    tf.keras.layers.Dropout(rate = 0.3),
    tf.keras.layers.Dense(units =10, activation = 'softmax')
])

model.compile(optimizer= tf.keras.optimizers.Adam(),
             loss= 'sparse_categorical_crossentropy',
             metrics = ['accuracy'])

model.summary()

history = model.fit(train_X, train_Y, epochs = 25, validation_split= 0.25)
plt.figure(figsize= (12,4))

plt.subplot(1,2,1)
plt.plot(history.history['loss'],'b-',label = 'loss')
plt.plot(history.history['val_loss'],'k--',label='val_loss')
plt.xlabel('Epoch')
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'],'g-',label = 'accuracy')
plt.plot(history.history['val_accuracy'],'r--',label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylim(0.7,1)
plt.legend()
plt.show()



model.evaluate(test_X, test_Y, verbose = 0)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import random

#다양한 변형된 이미지 
image_generator = ImageDataGenerator(
    rotation_range = 10,
    zoom_range= 0.10,
    shear_range = 0.5,
    width_shift_range= 0.10,
    height_shift_range = 0.10,
    horizontal_flip= True,
    vertical_flip= False
)

augment_size = 30000
randidx = np.random.randint(train_X.shape[0], size= augment_size)
x_augmented = train_X[randidx].copy()
y_augmented = train_Y[randidx].copy()

#flow()-> 보강된 이미지 생성
#bitch_size -> 한번에 생성할 이미지 생성
x_augmented = image_generator.flow(x_augmented, np.zeros(augment_size), batch_size = augment_size, shuffle= False).next()[0]


train_X = np.concatenate((train_X, x_augmented))
train_Y = np.concatenate((train_Y, y_augmented))

print(train_X.shape)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(input_shape = (28,28,1), kernel_size = (3,3) , filters= 32,
                           padding = 'same', activation='relu'),
    tf.keras.layers.Conv2D(kernel_size = (3,3), filters = 64, padding ='same', activation = 'relu'),
    tf.keras.layers.MaxPool2D(pool_size = (2,2)),
    tf.keras.layers.Dropout(rate = 0.5),
    tf.keras.layers.Conv2D(kernel_size = (3,3), filters = 128, padding = 'same', activation = 'relu'),
    tf.keras.layers.Conv2D(kernel_size = (3,3), filters = 256, padding = 'valid', activation = 'relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(rate = 0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units = 512, activation = 'relu'),
    tf.keras.layers.Dropout(rate = 0.5),
    tf.keras.layers.Dense(units = 256, activation = 'relu'),
    tf.keras.layers.Dropout(rate = 0.5),
    tf.keras.layers.Dense(units = 10, activation='softmax')
])

model.compile(optimizer =tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(train_X, train_Y, epochs = 25, validation_split = 0.25)

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], 'b-',label = 'loss')
plt.plot(history.history['val_loss'], 'r--',label = 'val_loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], 'g-', label = 'accuracy')
plt.plot(history.history['val_accuracy'], 'k--', label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylim(0.7,1)
plt.legend()

plt.show()

model.evaluate(test_X, test_Y, verbose = 0)