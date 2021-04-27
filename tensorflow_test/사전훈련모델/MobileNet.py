import tensorflow_hub as hub
import tensorflow as tf
mobile_net_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"
model = tf.keras.Sequential([
    hub.KerasLayer(handle = mobile_net_url, input_shape=(224,224,3), trainable = False)
])
model.summary()

import os
import pathlib
content_data_url = '/content/sample_data'
data_root_orig = tf.keras.utils.get_file('imagenetV2', 'https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-topimages.tar.gz',cache_dir=content_data_url, extract=True) 
print('he')
data_root = pathlib.Path('/Users/kimminyoung/Desktop/workspace/ML:DL/ML-DL/tensorflow_test/사전훈련모델/content/sample_data/datasets/iamgenetv2-topimages')
print(data_root)
for idx,item in enumerate(data_root.iterdir()):
    print(item)
    if idx == 9:
        break
    