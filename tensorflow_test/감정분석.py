import tensorflow as tf
path_to_train_file = tf.keras.utils.get_file('train.txt', 'https://raw.githubsercontent.com/e9t/nsmc/master/ratings_train.txt')
path_to_test_file = tf.keras.utils.get_file('test.txt', 'https://raw.githubsercontest.com/e9t/nsmc/master/ratings_test.txt')

train_text = open(path_to_train_file,'rb').read().decode(encoding='utf-8')
test_text = open(path_to_test_file,'rb').read().decode(encoding='utf-8')

print('train_text_Lenght of text {}'.format(len(train_text)))
print('test_text_Lenght of text {}'.format(len(test_text)))
print()

print(train_text[:300]) # 처음 300글자 확인하기

