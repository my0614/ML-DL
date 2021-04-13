import numpy as np
import tensorflow as tf
import math 

x = np.array([[1,1],[1,0],[0,1],[0,0]]) #입력값 x
y = np.array([[1],[0],[0],[0]]) #신경망 OR연산하기
w = tf.random.normal([2],0,1) # 가중치값
b = tf.random.normal([1],0,1) #편향값
b_x = 1 

#시그모이드함수
def sigmoid(x):
    return 1 / (1 + math.exp(-x))
for i in range(2000):
    error_sum = 0
    #4개의 입력값
    for j in range(4):
        output= sigmoid(np.sum(x[j]*w)+b_x *b)
        error = y[j][0] - output
        w = w + x[j] * 0.1 * error
        b = b + b_x * 0.1 * error
        error_sum += error
    if i % 200 == 199:
        print(i, error, output)
        
for i in range(4):
    print('X :',x[i], 'Y(기대값):',y[i], 'output(출력값) : ', sigmoid(np.sum(x[i] * w)+ b))
