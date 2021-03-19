# keras.model
## model이란?
딥러닝계산을 간편하게 하기 위한 추상적인 클래스를 의미한다.
여러함수와 변수의 묶음이라고 생각하면 된다.

model에서 많이 사용하는 구조는 Sequential이라는 구조이다.
순차적으로 뉴런과 뉴런이 합쳐진 단위인 레이어를 일직선으로 배치한것이다. 
### Dense
기본적인 레이어로써, 입력과 출력사이에 있는 뉴런과 뉴런들을 서로 연결해주는 레이어이다.

~~~
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

x = np.array([[1,1],[1,0],[0,1],[0,0]])
y = np.array([[0],[1],[1],[0]]) #xor 연산하기

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units= 2, activation='sigmoid', input_shape = (2,)),
    tf.keras.layers.Dense(units = 1, activation='sigmoid')
])
model.compile(optimizer = tf.keras.optimizers.SGD(lr=0.1), loss='mse')
model.summary()
~~~

- unit -> 레이어를 구성하는 뉴런의 수를 지정해준다.
- activation -> 활성화함수 사용하기 (현재 sigmoid함수 사용)
- input_shape -> 첫번째 레이어에서만 정의한다. 입력하는 차원의 수를 적어준다.

<b>model.compile<b>이 실제로 모델을 동작할 수 있도록 준비하는 명령이다.
~~~
model.compile(optimizer = tf.keras.optimizers.SGD(lr=0.1), loss='mse')
model.summary()
~~~

optimizer은 최적화함수 사용을 위해서 사용한다. 현재 사용한 최적화함수는 SGD(경사하강법)이다.

- loss -> 손실, error의 오차값을 뜻한다.
- mse -> 평균제곱오차로 기대출력에서 실제출력을 뺀 값에 제곱한 값의 평균을 구하는것이다.
