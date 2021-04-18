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
train_text = train_text.split(' ')
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
examples_per_epoch = len(text_as_int)
print(examples_per_epoch)
sentence_dataset = tf.data.Dataset.from_tensor_slices(text_as_int) #데이터셋 생성
sentence_dataset = sentence_dataset.batch(seq_length +1, drop_remainder= True)
for item in sentence_dataset.take(1):
    print(idx2word[item.numpy()])
    print(item.numpy())
def split_input_target(chunk):
    return [chunk[:-1], chunk[-1]]

train_dataset = sentence_dataset.map(split_input_target)
for x,y in train_dataset.take(1):
    print(idx2word[x.numpy()])
    print(x.numpy())
    print(idx2word[y.numpy()])
    print(y.numpy())

#데이터세트 shufle, batch 설정
BATCH_SIZE = 128
steps_per_epoch = examples_per_epoch
BUFFER_SIZE = 10000

train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder =True)
# 모델 정의하기
total_words = len(vocab)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(total_words, 100, input_length = seq_length),
    tf.keras.layers.LSTM(units= 100, return_sequences = True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(units = 100),
    tf.keras.layers.Dense(total_words, activation = 'softmax')
])
model.compile(optimizer = 'adam', loss = 'sparse_categorital_crossentropy', metrics = ['accuracy'])
model.summary()

#모델학습하기
from tensorflow.keras.preprocessing.sequence import pad_sequences

def testmodel(epoch, logs):
    if epoch % 5 != 0 and epoch != 49:
        return
    test_sentence - train_text[0]
    
    next_words = 100
    for _ in range(next_words):
        test_Text_X = test_sentence.split(' ')[-seq_length:]
        test_text_X = np.array([word2idx[c] if c in word2idx else word2idx['UNK'] for c in test_text_X])
        test_text_X = pad_sequences([test_text_X], maxlen=seq_length, padding = 'pre', value = word2idx['UNK'])
        
        output_idx = model.predict_classes(test_text_X)
        test_sentence += ' ' + idx2word[output_idx[0]]
        
    print()
    print('hi')
    print(test_sentence)
    print()
testmodelcb = tf.keras.callbacks.LambdaCallback(on_epoch_end = testmodel)
history = model.fit(train_dataset.repeat(), epochs = 50, steps_per_epoch = steps_per_epoch ,callbacks = [testmodelcb],verbose = 2)

        
    