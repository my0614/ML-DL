from __future__ import absolute_import, division, print_function, unicode_literals
try:
    %tensorflow_version 2.x
except Exception:
    pass
import tensorflow as tf


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

mpl.rcParams['figure.figsize'] = (8,6)
mpl.rcParams['axes.grid'] = False

zip_path = tf.keras.utils.get_file(origin = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',fname = 'jena_climate_2009_2016.csv.zip', extract = True)
csv_path, _ = os.path.splitext(zip_path)

df = pd.read_csv(csv_path)
df.head() #기본적으로 정보 5개 제공
# univariate -> 하나의 특성으로 예측한다.
# multivariate -> 여러개의 특성으로 예측한다.

def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size
    for i in range(start_index, end_index):
        indices = range(i-history_size, i)
        labels.append(dataset[i+target_size])
    return np.array(data),np.array(labels)

TRAIN_SPLIT = 300000
tf.random.set_seed(13) # 재현성을 위한 시드 설정

uni_data = df['T (degC)'] # 온도
uni_data.index = df['Date Time'] #시계열 시간
uni_data.head()

# uni_data들로 시각화
uni_data.plot(subplots = True)
uni_data = uni_data.values

uni_train_mean = uni_data[:TRAIN_SPLIT].mean() # 온도 평균 구하기
uni_train_std = uni_data[:TRAIN_SPLIT].std() # 온도 표준편차 구하기
uni_data = (uni_data-uni_train_mean) / uni_train_std
print('데이터 표준화',uni_data)
