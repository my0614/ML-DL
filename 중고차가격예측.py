import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.optimizers import Nadam
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import random
from pandas._config.config import describe_option
import numpy as np

df = pd.read_csv('./test-data.csv')

def data_nomalization(df):

    del df['New_Price'] # New_Price 삭제
    df = df.dropna(axis=0) # 결측값 있는 행 제거하기

    df['Name'] = df['Name'].str.split(' ').str[0] # 자동차 브랜드로 nomalization
    fuel = pd.get_dummies(df['Fuel_Type'])
    name = pd.get_dummies(df['Name'])

    owner = pd.get_dummies(df['Owner_Type'])
    Transmission = pd.get_dummies(df['Transmission'])
    mileage = pd.get_dummies(df['Mileage'].str.split(' ').str[1]) # Mileage로 nomalizatio

    # kilometor 원핫인코딩
    kilometor = pd.DataFrame()
    kilometor['Kilometers_Driven1'] = (df['Kilometers_Driven'] >= 0) & (df['Kilometers_Driven']  <= 50000)
    kilometor['Kilometers_Driven2'] = (df['Kilometers_Driven'] >= 50000) & (df['Kilometers_Driven']  <= 100000)
    kilometor['Kilometers_Driven3'] = (df['Kilometers_Driven'] >= 100000) & (df['Kilometers_Driven']  <= 200000)
    kilometor['Kilometers_Driven4'] = (df['Kilometers_Driven'] >= 200000) & (df['Kilometers_Driven']  <= 300000)
    kilometor['Kilometers_Driven5'] = (df['Kilometers_Driven'] >= 350000)


    # engine 원핫인코딩
    engine = pd.DataFrame()
    df["Engine"] = df["Engine"].str.split(' ').str[0]
    df['Engine'] = df["Engine"].astype(int)

    engine['engine1'] = (df['Engine'] >=0) & (df['Engine'] <= 1500)
    engine['engine2'] = (df['Engine'] >=1500) & (df['Engine'] <= 3000)
    engine['engine3'] = (df['Engine'] >=3000) & (df['Engine'] <= 4500)
    engine['engine4'] = (df['Engine'] >=4500)

    """
    power = pd.DataFrame()
    df['Power'] = df["Power"].str.split(' ').str[0]
    df.drop(df.loc[df['Power']=='null'].index, inplace=True)
    power['Power'] = df['Power']
    power['Power'] = power["Power"].astype(float)
    power['Power1'] = (power['Power'] >=0) & (power['Power'] <= 100)
    power['Power2'] = (power['Power'] >=100) & (power['Power'] <= 150)
    power['Power3'] = (power['Power'] >=150)
    """


    re = pd.concat([name,kilometor,fuel,Transmission,owner,mileage],axis = 1)
    for i in re.columns:
        re = re.astype({i:int})
    return re

if __name__ == '__main__':
    seed = 3
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    df_train = pd.read_csv('./train-data.csv')
    df_test = pd.read_csv('./test-data.csv')
    
    df_train = data_nomalization(df_train)
    df_test = data_nomalization(df_test)

    train_val = df_train.values
    test_val = df_test.values
    X = train_val[:,1:46].astype(float)
    Y = train_val[:,0].astype(float)

    X_test = test_val[:,1:46].astype(float)
    Y_test = test_val[:,0].astype(float)
    
    n_fold = 10
    skf = StratifiedKFold(n_splits = n_fold, shuffle = True, random_state = seed)
    accuracy2 = []
    i = 0
    for train, val in skf.split(X,Y):
        model = Sequential()
        # input_dim = dfframe 행개수
        model.add(Dense(1, input_dim = 45, activation = 'relu', kernel_initializer ='uniform'))
        model.add(BatchNormalization())
        model.add(Dense(24, activation = 'elu', kernel_initializer = 'uniform'))
        model.add(Dense(80, activation = 'elu', kernel_initializer = 'uniform'))
        model.add(Dense(128, activation = 'elu', kernel_initializer = 'uniform'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation = 'sigmoid', kernel_initializer = 'uniform'))
        model.compile(loss='binary_crossentropy', optimizer = Nadam(lr = 0.0005), metrics = ['accuracy'])
        earlyStopping_callback = EarlyStopping(monitor = 'val_loss', patience = 10)
        # callback 지정해주기
        #model.fit(X[train], Y[train], validation_data = (X[val], Y[val]), batch_size = 50, epochs = 1000, verbose = 1, callbakcs = [earlyStopping_callback])
        earlyStopping_callback = EarlyStopping(monitor = 'val_loss', patience = 10)
        model.fit(X[train], Y[train], validation_data = (X[val], Y[val]), batch_size = 50, epochs = 1000, verbose = 1, callbacks=[earlyStopping_callback])
        model.save(f'k_fold_{i}.pt')
        k_accuracy = '%.4f' %(model.evaluate(X_test, Y_test)[1])
        print(k_accuracy)
        print("=" * 100)
        #예측하기
        print(X_test.shape)
        predictions = model.predict(X_test).reshape(X_test.shape[0])
        predictions = np.where(predictions>0.5,1,0)
        accuracy2.append(k_accuracy)
        print(accuracy2)
        print(predictions)
        i = i+1 
        
