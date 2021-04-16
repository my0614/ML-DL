import tensorflow as tf
path_to_file = tf.keras.utils.get_file('text.txt','http://bit.ly/2Mc3SOV')

train_text = open(path_to_file, 'rb').read().decode(encoding = 'utf-8')
print('Length  of text : {} character'.format(len(train_text)))
print()

print(train_text[:100])
import re

def clean_str(string):
    string = re.sub(r"[^가-힣A-Za-z-9(),!?\'\']", " ", string)      
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
#train_text = train_text.split('\n')
train_text = [clean_str(sentence) for sentence in train_text]
train_text_X = []
for sentence in train_text:
    train_text_X.extend(sentence.split(' '))
    train_text_X.append('\n')
    
train_text_X = [word for word in train_text_X if word != ''] # 단어가 없을때까지
print(train_text_X[:20])

