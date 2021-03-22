import tensorflow as tf
import tnesorflow as tf

a  = tf.Variable(random.random())
b = tf.Variable(random.random())

#잔차의 제곱의 평균을 반환하는 함수
def compute_loss():
    y_pred= a * X + b #1차원함수
    loss = tf.reduce_mean((Y-ypred)** 2) # Y는 기대값, y_pred는 실제값이다. 현재 오차의 제곱을 해준다.
    return loss 
