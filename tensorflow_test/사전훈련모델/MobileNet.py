import tensorflow_hub as hub
import tensorflow as tf
import cv2
import numpy as np
import os
import pathlib
import PIL.Image as Image
import matplotlib.pyplot as plt
import random
import pandas as pd


mobile_net_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"
model = tf.keras.Sequential([
    hub.KerasLayer(handle = mobile_net_url, input_shape=(224,224,3), trainable = False)
])
model.summary()

content_data_url = '/content/sample_data'
data_root_orig = tf.keras.utils.get_file('imagenetV2', 'https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-topimages.tar.gz',cache_dir=content_data_url, extract=True) 
print('he')
data_root = pathlib.Path('/Users/kimminyoung/Desktop/workspace/ML:DL/ML-DL/tensorflow_test/사전훈련모델/content/sample_data/datasets/iamgenetv2-topimages')
print(data_root)
for idx,item in enumerate(data_root.iterdir()):
    print(item)
    if idx == 9:
        break
label_file = tf.keras.utils.get_file('label','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
label_text = None
with open(label_file, 'r') as f:
    label_text = f.read().split('\n')[:-1]
print(len(label_text))
print(label_text[:10])
print(label_text[-10:])

all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)

image_count = len(all_image_paths)
print('image_count:', image_count)

plt.figure(figsize = (12,12))
for c in range(9):
    image_path = random.choice(all_image_paths)
    plt.subplot(3,3,c+1)
    plt.imshow(plt.imread(image_path))
    idx = int(image_path.split('/')[-2]) + 1
    plt.title(str(idx) + ',' + label_text[idx])
    plt.axis('off')
plt.show()
# 예측 퍼센트 

top_1 = 0
top_5 = 0

for image_path in all_image_paths:
    img = cv2.imread(image_path)
    img = cv2.resize(img, dsize =(224,224))
    img = img/255.0
    img = np.expand_dims(img,axis = 0)
    top_5_predict = model.predict(img)[0].argsort()[::-1][:5]
    idx = int(image_path.split('/')[-2])+1
    if idx in top_5_predict:
        top_5 += 1
        if top_5_predict[0] == idx:
            top_1 += 1
print('Top-5 correctness:', top_5 / len(all_image_paths) * 100, '%')
print('Top-1 correctness:', top_1 / len(all_image_paths) * 100, '%')

plt.figure(figsize = (16,16))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis = 0)

for c in range(3):
    image_path = random.choice(all_image_paths)
    
    plt.subplot(10,2,c*2+1)
    plt.imshow(plt.imread(image_path))
    idx = int(image_path.split('/')[-2]) + 1
    plt.title(str(idx) + ',' + label_text[idx])
    plt.axis('off')
    #예측값
    plt.subplot(10,2,c*2+2)
    img = cv2.imread(image_path)
    img = cv2.resize(img, dsize=(224,224))
    img = img/255.0
    img = np.expand_dims(img, axis = 0)
    #MobileNet으로 예측하기
    logits = model.predict(img)[0]
    prediction = softmax(logits)
    
    #높은 예측값 뽑기
    top_5_predict = prediction.argsort()[::-1][:5] # 5개뽑기
    labels = [label_text[index] for index in top_5_predict]
    color = ['gray'] * 5
    if idx in top_5_predict:
        color[top_5_predict.tolist().index(idx)] = 'green'
    color = color[::-1]
    plt.barh(range(5), prediction[top_5_predict][::-1]*100, color = color)
    plt.yticks(range(5), labels[::-1])

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
