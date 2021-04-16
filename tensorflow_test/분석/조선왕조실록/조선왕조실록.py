import tensorflow as tf
path_to_file = tf.keras.utils.get_file('text.txt','http://bit.ly/2Mc3SOV')

train_text = open(path_to_file, 'rb').read().decode(encoding = 'utf-8')
print('Length  of text : {} character'.format(len(train_text)))
print()

print(train_text[:100])
import re

def clean_str(string):
    string = re.sub(r"[^가-힣A-Za-z0-9(),!?\'\']", " ", string)      
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", "", string) 
    string = re.sub(r"\)", "", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\'{2,}", "\'", string)
    string = re.sub(r"\'","", string)
   
    return string
#
train_text = [clean_str(sentence) for sentence in train_text]

train_text_X = []
for sentence in train_text:
    train_text_X.extend(sentence.split(' '))
    #train_text_X.append('\n')
    
train_text_X = [word for word in train_text_X if word != ''] # 단어가 없을때까지
print(train_text_X[:20])

import numpy as np
vocab = sorted(set(train_text_X)) 
vocab.append('UNK')
print('{} unique word'.format(len(vocab)))

#각각 단어들을 단어들과 숫자를 매칭해줍니다.
word2idx = {u:i for i, u in enumerate(vocab)}
idx2word= np.array(vocab)

text_as_int = np.array([word2idx[c] for c in train_text_X]) #단어에 대한 숫자매핑
print('{')

# 단어 시퀸스정하기
for word,_ in zip(word2idx, range(10)):
    print(' {:4s}: {:3d},'.format(repr(word), word2idx[word]))
print(' ...\n}')
print('index of UNK: {}'.format(word2idx['UNK']))

print(train_text_X[:20])
print(text_as_int[:20])

seq_length = 25 # 단어 시퀸스 최대길이
examples_per_epoch = len(text_as_int)
print(examples_per_epoch)
sentence_dataset = tf.data.Dataset.from_tensor_slices(text_as_int) #데이터셋 생성
sentence_dataset = sentence_dataset.batch(seq_length +1, drop_remainder= True)
for item in sentence_dataset.take(1):
    print(idx2word[item.numpy()])
    print(item.numpy())

#다음에 나올 글자 예측하기
def split_input_target(chunk):
    return [chunk[:-1], chunk[-1]]

train_dataset = sentence_dataset.map(split_input_target)
for x,y in train_dataset.take(1):
    print(idx2word[x.numpy()])
    print(x.numpy())
    print(idx2word[y.numpy()])
    print(y.numpy())