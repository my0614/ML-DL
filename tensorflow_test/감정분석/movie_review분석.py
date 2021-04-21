
from bs4 import BeautifulSoup
import requests
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

#5페이지만



import numpy as np
import re
import tensorflow as tf

#데이터다운로드
path_to_train_file = tf.keras.utils.get_file('train_txt', 'https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt')
path_to_test_file = tf.keras.utils.get_file('test_txt', 'https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt')


#파일열기
train_text = open(path_to_train_file, 'rb').read().decode(encoding ='utf-8')
test_text = open(path_to_test_file,'rb').read().decode(encoding='utf-8')

print('Length of text : {} characters'.format(len(train_text)))
print('Length of text : {} characters'.format(len(test_text)))
print()

print(train_text[:1000]) #처음 1000글자 

#label가지고 오기 (부정 : 0, 긍정 : 1)
train_Y = np.array([[int(row.split('\t')[2])] for row in train_text.split('\n')[1:] if row.count('\t') > 0]) #if row.count('\t') > 0 이면 .txt 파일에 내용이 있는지 비교
test_Y = np.array([[int(row.split('\t')[2])] for row in test_text.split('\n')[1:] if row.count('\t') > 0]) #if row.count('\t') > 0 이면 .txt 파일에 내용이 있는지 비교
print(train_Y.shape, test_Y.shape)
print(train_Y[:10])

#정규화함수
def clean_str(string):
    string = re.sub(r"[^가-힣A-Za-z-9(),!?\'\']", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\'{2,}", "\'", string)
    string = re.sub(r"\'","", string)

    return string.lower()


train_text_X = [row.split('\t')[1] for row in train_text.split('\n')[1:] if row.count('\t') > 0]
train_text_X = [clean_str(sentence) for sentence in train_text_X] # 정규화해주기
sentences = [sentence.split(' ') for sentence in train_text_X] #문장을 띄어쓰기로 자르기
for i in range(5):
    print(sentences[i])

sentence_len = [len(sentence) for sentence in sentences]
sentence_len.sort() #문자열길이 정렬하기
plt.plot(sentence_len)
plt.show()

#print(sum([int(i<=25) for i in sentence_len]))

#단어길이 줄이기(정제)
sentences_new = []
for sentence in sentences:
    sentences_new.append([word[:5] for word in sentence][:25])
sentences = sentences_new
for i in range(5):
    print(sentences[i])

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words = 20000)
tokenizer.fit_on_texts(sentences)
train_X = tokenizer.texts_to_sequences(sentences)
train_X = pad_sequences(train_X, padding='post')

print(train_X[:5])

print(tokenizer.index_word[19999])
print(tokenizer.index_word[20000])

temp = tokenizer.texts_to_sequences(['#$#$', '매력으로','영화네유 ','연기가']) #문자 입력받기
print(temp)
temp = pad_sequences(temp,padding = 'post') #패딩설정하기
print(temp)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(20000, 300, input_length = 25), #각 문장에 들어있는 25개의 단어를 300의 임베딩 벡터로 변환합니다.
    tf.keras.layers.LSTM(units = 50),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer ='adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])
model.summary()

history = model.fit(train_X, train_Y, epochs = 5, batch_size = 128, validation_split = 0.2)

plt.figure(figsize = (12,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'],'b-',label = 'loss')
plt.plot(history.history['val_loss'],'r--',label= 'val_loss')
plt.xlabel('Epoch')
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'],'g-', label='accuracy')
plt.plot(history.history['val_accuracy'],'k--', label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylim(0.7, 1)
plt.legend()

plt.show()

test_text_X = [row.split('\t')[1] for row in test_text.split('\n')[1:] if row.count('\t') > 0]
test_text_X = [clean_str(sentence) for sentence in test_text_X] # 정규화해주기
sentences = [sentence.split(' ') for sentence in test_text_X]
sentences_new = []
for sentence in sentences:
    sentences_new.append([word[:5] for word in sentence][:25])
sentences = sentences_new

test_X = tokenizer.texts_to_sequences(sentences)
test_X = pad_sequences(test_X, padding = 'post')

model.evaluate(test_X, test_Y, verbose = 0)
class person(object):
    def __init__(self,name):
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return "'"+self.name+"'"

movie_name = []
review = []
def find_key(value1):
    for key, value in movie.items():
        if value == value1:
            return key
            
for page in range(1,22):
    #print('page : ',page)
    source = requests.get("https://movie.naver.com/movie/point/af/list.nhn?&page=%d" % page).text
    soup = BeautifulSoup(source, "html.parser")
    hotkeys = soup.select("a.movie.color_b") #class가지고 오기
    text_value = soup.select(".title")
    for i in range(10):
        a=  text_value[i].text.split('\n')
        movie_name.append(a[1])
        review.append(a[5])

state = []
for i in range(len(review_list)):
    test_sentence =  review[i]# 원하는문장 넣어보기!!
    test_sentence = test_sentence.split(' ')
    test_sentences =[]
    now_sentence= []
    for word in test_sentence:
        now_sentence.append(word)
        test_sentences.append(now_sentence[:])
    test_X_1 = tokenizer.texts_to_sequences(test_sentences)
    test_X_1 = pad_sequences(test_X_1, padding ='post', maxlen = 25) #문장 최대길이 25로 맞춤

    prediction = model.predict(test_X_1)
    for idx, sentence in enumerate(test_sentences):
            pass
    #print(sentence)
    #print(prediction[idx][1])
    if prediction[idx][1] > 0.5:
        state.append('1')
    else:
        state.append('0')
number = []
for i in range(len(state)):
    number.append(i) 
print(len(movie_name), len(state))
import pandas as pd
import numpy as np

df = pd.DataFrame(movie_name,columns =['영화 이름'])
df.insert(1,'리뷰상태', state)
name = input()
print('영화이름 : ', name)
#원하는 영화 리뷰보기
print(len(df[df['영화 이름'] == name]))
print('긍정적인 리뷰의 개수는',len(df[(df['리뷰상태'] == '1')& (df['영화 이름'] == name)]))
print('부정적인 리뷰의 개수는',len(df[(df['리뷰상태'] == '0')& (df['영화 이름'] == name)]))
