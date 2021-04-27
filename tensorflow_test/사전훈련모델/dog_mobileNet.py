
#강아지 종 개수 확인

label_text= pd.read_csv('./dog-breed-identification/labels.csv')
print(label_text.head())
print(label_text['breed'].nunique())

#강아지 라벨과 이미지 같이 출력


plt.figure(figsize = (12,12))
for c in range(9):
    image_id = label_text.loc[c,'id']
    plt.subplot(3,3,c+1)
    plt.imshow(plt.imread('./dog-breed-identification/train/' + image_id + '.jpg'))
    plt.title(str(c) +',' + label_text.loc[c, 'breed'])
    plt.axis('off')
plt.show()
#MobileNetV2 불러오기
from tensorflow.keras.applications import MobileNetV2
mobilev2 = MobileNetV2()

#훈련데이터 메모리에 로드
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.applications import MobileNetV2
mobilev2 = MobileNetV2()
#layer 전체 true로 변환 하지만 마지막 Dense는 제외함
for layer in mobilev2.layers[:-1]:
    layer.trainable = True
#레이어에 커널이 있는지 확인하고 랜덤변수로 초기화    
for layer in mobilev2.layers[:-1]:
    if 'kernel' in layer.__dict__:
        kernel_shape = np.array(layer.get_weights()).shape
        layer.set_weights(tf.random.normal(kernel_shape,0,1))
train_X = []
for i in range(len(label_text)):
    img = cv2.imread('./dog-breed-identification/train/' + label_text['id'][i] +'.jpg')
    #print(img)
    img = cv2.resize(img, dsize =(224,224))
    img = img /255.0
    train_X.append(img)
train_X = np.array(train_X)
print(train_X.shape)
print(train_X.size * train_X.itemsize, ' bytes')
