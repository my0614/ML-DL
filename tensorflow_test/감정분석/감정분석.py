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

