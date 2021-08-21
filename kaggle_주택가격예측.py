import pandas as pd
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import StratifiedKFold
import numpy as np
from tensorflow.keras.optimizers import Nadam
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import random


def data_nomalization(df):
    df = df[['MSSubClass',"Alley","HouseStyle","OverallQual","OverallCond","YearBuilt","Foundation","CentralAir","KitchenQual","GarageQual","GarageCond","YrSold","SaleCondition"]]

    #data nomalization
    df['CentralAir'] = (df['CentralAir'] == 'Y')
    df.astype({'MSSubClass':int}) # 자료형 int로 변환
    df['MSSubClass1'] = (df['MSSubClass'] >= 20) & (df['MSSubClass'] <= 40)
    df['MSSubClass2'] = (df['MSSubClass'] > 40) & (df['MSSubClass'] <= 70)
    del df['MSSubClass']
    df['Alley'] = (df['Alley'].fillna(0))
    df['Alley'] = (df['Alley']!= 0)
    df['HouseStyle']= df.HouseStyle.str.extract('(\d+)')
    df['HouseStyle'].fillna(random.choice(['1','2']), inplace = True) # 랜덤으로 Embarked 정해주기
    df['HouseStyle1floor']= (df['HouseStyle'] == '1')
    df['HouseStyle2floor']= (df['HouseStyle'] == '2')
    del df['HouseStyle']

    df['OverallQual'] = (df['OverallQual']>=1) & (df['OverallQual'] <= 5)
    df['OverallQual2'] = (df['OverallQual']>=6) & (df['OverallQual'] <= 10)
    del df['OverallQual']


    df['YearBuilt1800'] =  (df['YearBuilt']>= 1800) & (df['YearBuilt'] <=1899)
    df['YearBuilt1900'] =  (df['YearBuilt']>= 1900) & (df['YearBuilt'] <=1999)
    df['YearBuilt2000'] =  (df['YearBuilt']>= 2000) & (df['YearBuilt'] <=2099)
    del df['YearBuilt']

    df['OverallCond1'] = (df['OverallCond']>=1) & (df['OverallCond'] <= 5)
    df['OverallCond2'] = (df['OverallCond']>=6) & (df['OverallCond'] <= 10)
    del df['OverallCond']

    df['KitchenQual1'] = (df['KitchenQual'] == 'Ex') | (df['KitchenQual'] == 'Gd') | (df['KitchenQual'] == 'TA')
    df['KitchenQual0'] = (df['KitchenQual'] == 'Fa') | (df['KitchenQual'] == 'Po') | (df['KitchenQual'] == 'NA')
    del df['KitchenQual']
    del df['GarageCond']

    df['GarageQual1'] = (df['GarageQual'] == 'Ex') | (df['GarageQual'] == 'Gd') | (df['GarageQual'] == 'TA')
    df['GarageQual0'] = (df['GarageQual'] == 'Fa') | (df['GarageQual'] == 'Po') | (df['GarageQual'] == 'NA')
    del df['GarageQual']
    del df['YrSold']
    foundation= pd.get_dummies(df['Foundation'])
    del df['Foundation']
    del df['SaleCondition']
    df
    for col in df.columns:
        df = df.astype({col:int})
    return df

if __name__ == '__main__':
    seed = 3
    np.random.seed(seed)
    tf.random.set_seed(seed)

    df_train = pd.read_csv("./test.csv")
    df_test = pd.read_csv("./test.csv")
    sample = pd.read_csv("./sample_submission.csv")

    df_train = data_nomalization(df_train)
    df_test = data_nomalization(df_test)

    train_val = df_train.values 
    test_val = df_test.values

    X = train_val[:,1:15].astype(float)
    Y = train_val[:,0].astype(float)


    X_test = test_val[:,1:15].astype(float)
    Y_test = test_val[:,0].astype(float)

    n_fold = 10
    skf = StratifiedKFold(n_splits = n_fold, shuffle = True, random_state = seed)
    accuracy2 = []
    i = 0
    for train,val in skf.split(X,Y):
        #모델 생성
        model = Sequential()
        """
        model.add(Dense(1, input_dim = 14, activation = 'relu'))
        model.add(Dense(24, activation = 'relu'))
        model.add(Dense(100, activation = 'relu'))
        model.add(Dense(150, activation = 'relu'))
        model.add(Dense(1, activation = 'sigmoid'))
        
        """
        #모델구축
        model.add(Dense(32, input_dim = 14, activation = 'relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1)) # output = 1
        model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['accuracy']) 
        earlyStopping_callback = EarlyStopping(monitor = 'val_loss', patience = 10)

        earlyStopping_callback = EarlyStopping(monitor = 'val_loss', patience = 10)
        model.fit(X[train], Y[train], validation_data = (X[val], Y[val]), batch_size = 33, epochs = 100, verbose = 1, callbacks=[earlyStopping_callback])
        #model.save(f'k_fold_{i}.pt')
        k_accuracy = '%.4f' %(model.evaluate(X_test, Y_test)[1])
        print(k_accuracy)
        print("=" * 100)
        #예측하기
        print(X_test.shape)
        predictions = model.predict(X_test).reshape(X_test.shape[0])
        accuracy2.append(k_accuracy)
        #sample.to_csv(f'./kim_submission{i}.csv',index=False) #csv 파일저장
        i = i+1
    print(accuracy2)
