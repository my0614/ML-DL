import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

X = [0.3,-0.78,1.26,0.03,1.11,0.24,-0.24,-0.47,-0.77,-0.37,-0.85,-0.41,-0.27,0.02,-0.76,2.66]
Y = [12.27,14.44,11.87,18.75,17.52,16.37,19.78,19.51,12.65,14.74,10.72,21.94,12.83,15.51,17.14,14.42]
print(len(X),len(Y))

#a,b,c 랜덤값으로 초기화
a = tf.Variable(random.random())
b = tf.Variable(random.random())
c = tf.Variable(random.random())

def compute_loss():
    y_pred=a*X*X + b * X+c # 2차함수식
    loss = tf.reduce_mean((Y-y_pred) ** 2) #전차의 제곱 평균
    return loss

optimizer = tf.keras.optimizers.Adam(lr=0.07) #활성화 함수로 Adam사용
for i in range(1000):
    optimizer.minimize(compute_loss, var_list = [a,b,c])
    if i % 100 == 99:
        print(i, 'a:', a.numpy(), 'b:', b.numpy(), 'c:', c.numpy(), 'loss:', compute_loss().numpy())

line_x = np.arange(min(X),max(X), 0.01)
line_y = a * line_x * line_x + b * line_x + c

plt.plot(line_x, line_y)
plt.plot(X,Y,'bo')
plt.xlabel('population Growth rate(%)')
plt.xlabel('Elderly population rate(%)')
plt.show()
