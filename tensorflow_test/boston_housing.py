from tensorflow.keras.datasets import boston_housing
import tensorflow as tf
(train_X, train_Y), (test_X, test_Y) = boston_housing.load_data()
print(len(train_X), len(train_Y))
print(train_X[0])
print(train_Y[0])

x_mean = train_X.mean(axis = 0)
x_std = train_X.std(axis = 0)
train_X -= x_mean
train_X /= x_std
test_X -= x_mean
test_X /= x_std

y_mean = train_Y.mean(axis = 0)
y_std = train_Y.std(axis = 0)

train_Y -= y_mean
train_Y /= y_std
test_Y -= y_mean
test_Y /= y_std

print(train_X[0])
print(train_Y[0])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units = 50, activation = 'relu', input_shape = (13,)),
    tf.keras.layers.Dense(units = 35, activation = 'relu'),
    tf.keras.layers.Dense(units = 21, activation = 'relu'),
    tf.keras.layers.Dense(units= 1) #출력은 1개
])
model.compile(optimizer = tf.keras.optimizers.Adam(lr = 0.07), loss = 'mse')
model.summary()
history = model.fit(train_X, train_Y, epochs = 25, batch_size = 32, validation_split = 0.25)


pred_Y = model.predict(test_X)
plt.figure(figsize = (5,5))
plt.plot(test_Y, pred_Y, 'b.')
plt.axis([min(test_Y) , max(test_Y), min(test_Y), max(test_Y)])

plt.plot([min(test_Y) , max(test_Y)],[ min(test_Y), max(test_Y)], ls = '--', c = ".3")
plt.xlabel('test_Y')
plt.ylabel('test_Y')
plt.show()
