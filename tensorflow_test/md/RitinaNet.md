# RetinaNet
예측하기 어려운 예제에 집중하도록 Focal Loss를 제안합니다. ResNet과 FPN을 활용하여 구축한 one-state 모델 RetinaNet은 focal loss를 사용하여 정확도를 높였다.
- 정확도가 낮은 원인은 객체와 배경의 클래스 불균형때문이다.

two-stage 기법은 휴리스틱 샘플링 또는 OHEM 기법을 클래스 불균형 문제를 해결하는데 사용합니다. hard negative sample을 골라서 mini-batch를 구성한 뒤에 모델을 학습시키고 False negative으로 인해 정확도가 높아집니다.

loss function을 수정하여, 쉬운 예제에 0에 가까운 loss를 부여하고, 예측하기 어려운 예제에 기존보다 높은 loss을 부여합니다. 


## RetinaNet
RetinaNet은 ResNet-FPN 2개와 sub-network를 사용하여 만든 신경망입니다. 첫번째 sub-network는 object-classfication을 수행하고, 두번째 sub-network는 bounding box regression을 수행합니다. 
