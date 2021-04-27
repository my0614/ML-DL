import jamotools
import tensorflow as tf
import numpy as np

path_to_file = tf.keras.utils.get_file('text.txt','http://bit.ly/2Mc3SOV')

train_text = open(path_to_file, 'rb').read().decode(encoding = 'utf-8')
s = train_text[:100]
print(s)

s_split = jamotools.split_syllables(s) # 자모를 나누기
print(s_split)
s2 = jamotools.join_jamos(s_split)
print(s2)
print(s == s2)

train_text_X = jamotools.split_syllables(train_text)
vocab = sorted(set(train_text_X)) # 정렬뒤, 중복값 제거
vocab.append('UNK')
print('{} unique characters'.format(len(vocab)))

char2idx = {u:i for i,u in enumerate(vocab)}
idx2char = np.array(vocab)

test_as_int = np.array([char2idx[c] for c in train_text_X])
print('{')
for char,_ in zip(char2idx, range(10)):
    print(' {:4s}: {:3d},'.format(repr(char), char2idx[char]))
print(' ...\n')
print('index of UNK: {}'.format(char2idx['UNK']))

#출력해보기
print(train_text_X[:20])
print(test_as_int[:20])

seq_length = 80
examples_per_epoch = len(test_as_int)
char_dataset = tf.data.Dataset.from_tensor_slices(test_as_int)

char_dataset = char_dataset.batch(seq_length+1, drop_remainder = True)
for item in char_dataset.take(1):
    print(idx2char[item.numpy()])
    print(item.numpy())
def split_input_target(chunk):
    return [chunk[:-1], chunk[-1]]

train_dataset = char_dataset.map(split_input_target)
for x,y in train_dataset.take(1):
    print(idx2char[x.numpy()])
    print(x.numpy())
    print(idx2char[y.numpy()])

BATCH_SIZE = 256
steps_per_epoch = examples_per_epoch
BUFFER_SIZE = 10000
