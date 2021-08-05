import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

dir_data = './datasets'
name_label = 'train-labels.tif'
name_input = 'train-volume.tif'
img_label = Image.open(os.path.join(dir_data,name_label)) # train_labels.tif파일열기
img_input = Image.open(os.path.join(dir_data,name_input)) # train_volume.tif파일열기

ny, nx = img_label.size
nframe = img_label.n_frames

nframe_train = 24
nframe_val = 3
nframe_test = 3

dir_save_train = os.path.join(dir_data, 'train')
dir_save_val = os.path.join(dir_data, 'val')
dir_save_test = os.path.join(dir_data, 'test')

if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)

if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)

if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test)

#train, val, test

#
id_frame = np.arange(nframe)
np.random.shuffle(id_frame)

offset_nframe = 0

for i in range(nframe_train):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_train, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_train, 'input_%03d.npy'%i), input_)



offset_nframe += nframe_train # 프레임수 더하기

# val npy파일 만들기
for i in range(nframe_val):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_val, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_val, 'input_%03d.npy'%i), input_)

offset_nframe += nframe_val # 프레임수 더하기

# val npy파일 만들기
for i in range(nframe_test):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_test, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_test, 'input_%03d.npy'%i), input_)

#label과 input 사진 보기
plt.subplot(121)
plt.imshow(label_,cmap='gray')
plt.title('label')

plt.subplot(122)
plt.imshow(input_, cmap='gray')
plt.title('input')
plt.show()
