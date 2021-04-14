import numpy as np
import tensorflow as tf

X = []
Y = []

for i in range(6):
    lst = list(range(i,i+4))

    X.append(list(map(lambda c : [c/10], lst)))
    Y.append((i+4)/10)
X = np.array(X)
Y = np.array(Y)
for i in range(len(X)):
    print(X[i], Y[i])
    
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(units = 10, return_sequences = False, input_shape = [4,1]),
    tf.keras.layers.Dense(1) #별도의 활성화함수가 없다.
])

model.compile(optimizer = 'adam', loss= 'mse')
model.summary()

model.fit(X,Y, epochs= 100, verbose = 0)
print(model.predict(np.array([[[0.3],[0.4],[0.5],[0.6]]])))