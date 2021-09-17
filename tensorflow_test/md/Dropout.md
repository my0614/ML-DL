# 드롭아웃

## 드롭아웃이란?

overfitting을 줄이기 위한 방법 중 하나이다. hidden layer의 일부 유닛이 동작하지 않게 하여 overfitting을 막는다.

## OverFitting이란?

모델이 학습 데이터를 지나치게 학습하여 실험데이터에서는 결과가 좋지 않은 경우을 의미한다.

## OverFittig 해결

1. 데이터의 수를 늘려 이러한 overfitting을 줄일 수 있다.
2. Regularization을 사용하여 학습데이터에 지나치게 의존하지 않도록 패널티를 준다.
이러한 방법에 Dropout과 Regularization이 있다.

## Dropout

![2536663857F247DA11](https://i.imgur.com/UAyoVvf.png)

네트워크의 유닛의 일부만 작동하고 일부는 작동하지 않도록 한다. model combination을 하게 되면 모델의 학습을 시키거나 서로 다른 구조의 학습성능을 개선할 수 있다. 하지만 여러개의 네트워크를 훈련시켜도 사용시에 연산시간이
많이 사용된다거나 속도가 느려지는 문제가 있다.

이러한 문제를 해결하기 위해 개발된 Dropout은 여러개의 모델을 만들지 않고 모델결합이 여러개의 형태를 가지게 한다. Dropout 사용법은 일부 뉴런이 동작하지 않도록 하는것인데

- 각 뉴런이 존재할 확률 p를 가중치 w와 곱해주는 형태가 되어 존재할 확률이 0인 경우 동작하지 않도록 한다.

## Dropout Nets
기본적인 Neural Nets과 같이 SGD을 사용하여 학습을 할 수 있다. 

p값에 따른 train error와 test error가 있다. 0\~0.3 정도에서는 error값이 증가하는데
dropout을 너무 많이 시키면 이러한 결과가 발생할 수 있다. 또한 0.9\~1.0까지도 error가 올라가는데 1.0값은 Dropout을 안한것과 같기 때문에 overfitting이 일어났다고 이야기 할 수 있다. 확률값을 잘 사용해야 한다.

~~미녕이바보~~

## Dropconnect

![2139423857FB6F602B](https://i.imgur.com/4qIdNM3.png)

Dropout을 변형한 방식중 하나로 Dropout이 unit을 drop한것이라면 Dropconnect은 다른 레이어간의 유닛사이의 연결을 Drop하는 것이다. 두 유닛 간의 연결을 drop함으로써 한 모델에서 여러모델을 훈련시키는 것과 같다.

