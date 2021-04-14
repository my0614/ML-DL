import tensorflow as tf
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))
x = 1
y = 0
w = tf.random.normal([1],0,1)
for i in range(1000):
    output = sigmoid(x*w)
    error = y - output
    w = w + x * 0.1 * error
    
    if i % 100 == 99:
        print(i, error, output)
    
