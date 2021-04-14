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


