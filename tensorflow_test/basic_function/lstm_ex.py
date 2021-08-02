from numpy import array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping
"""
# 1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5],[4,5,6],[5,6,7],
           [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],
            [20,30,40], [30,40,50], [40,50,60]])

y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])
"""

size = 5 
print('x shape : ', x.shape) 
print('y shape : ', y.shape)

print(x)
print('-------x reshape-----------')
x = x.reshape((x.shape[0], x.shape[1], 1)) # (4,3,1) reshape 전체 곱 수 같아야 4*3=4*3*1
print('x shape : ', x.shape)
print(x)

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size  +1):
        subset = seq[i:(i+size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a,size)
print(dataset)

x_train = dataset[:,0:-1]
y_train = dataset[:,-1]

print(x_train.shape)
print(y_train.shape)



model = Sequential()
model.add(Dense(33, activation = 'relu', input_shape=(4,)))
model.add(Dense(5))
model.add(Dense(1))

model.summary()

model.compile(optimizer = 'adam', loss = 'mse')
model.fit(x_train,y_train,epochs = 100, batch_size = 1, verbose = 2)
x_input = array([8,9,10,11]) # input값
x_input = x_input.reshape(1,4)

yhat = model.predict(x_input)
print(yhat)
