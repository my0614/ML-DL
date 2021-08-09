from pandas._config.config import describe_option
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import StratifiedKFold

import pandas as pd
import numpy as np
from tensorflow.keras.optimizers import Nadam
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import random

def age_normalization(age):
    age_nor = 0
    if age >= 65:
        age_nor = 6
    elif age>= 50:
        age_nor = 5
    elif age>= 30:
        age_nor = 4
    elif age >= 19:
        age_nor = 3
    elif age >= 13:
        age_nor = 2
    elif age >= 5:
        age_nor = 1
    else:
        age_nor = 0


# 결측값을 평균으로 맞춰주기
# pd.isnull(df).sum() -> 결측값개수 알기
embarked = ['S','C','Q'] # Embarked 종류
def non_df(df):
    df['Age'].fillna(df['Age'].mean(), inplace = True) # 나이 결측치 평균으로 맞추기
    df['Fare'].fillna(df['Fare'].mean(), inplace = True) # Cabin 결측치 평균으로 맞추기
    df['Embarked'].fillna(random.choice(embarked), inplace = True) # 랜덤으로 Embarked 정해주기
    df = df.drop(columns= 'Ticket') # 필요없는 데이터 삭제
    df = df.drop(columns = 'Cabin')

    return df

def df_normalization(df):

    df = df.set_index("PassengerId")
    #df=df['Sex'] =='male' = 0 # 남자 0으로 변환
    #df= df['Sex'] =='female' = 1 # 여자 1로 변환
    df['Sex'] = (df['Sex'] == 'female') #여자가 아니면
    
    # Pclass 별로 나누기
    df['Pclass1'] = (df['Pclass'] == 1)
    df['Pclass2'] = (df['Pclass'] == 2) 
    df['Pclass3'] = (df['Pclass'] == 3)
    df = df.drop(columns='Pclass') # pclass 행 삭제

    df['Age'] = df['Age'].apply(lambda x : age_normalization)
    df['Age1']=(df['Age']==0)
    df['Age2']=(df['Age']==1)
    df['Age3']=(df['Age']==2)
    df['Age4']=(df['Age']==3)
    df['Age5']=(df['Age']==4)
    df['Age6']=(df['Age']==5)
    df['Age7']=(df['Age']==6)
    df = df.drop(columns='Age') # Age행 삭제하기

    df['Em_C'] = (df['Embarked'] == 'C')
    df['Em_S'] = (df['Embarked'] == 'S')
    df['Em_Q'] = (df['Embarked'] == 'Q')
    df = df.drop(columns='Embarked') # Embarked 행 삭제하기
    #이름에서 호칭추출하기
    df['family'] = df['SibSp'] + df['Parch']

    df['1_3'] = (df['family'] >= 1) & (df['family'] <= 3)
    df['only'] = df['family'] == 0
    df['many'] = df['family'] >= 4
    df = df.drop(columns = ['family','SibSp','Parch'])


    df['Fare_1'] = (df['Fare'] >= 0) & (df['Fare'] < 10)
    df['Fare_2'] = (df['Fare'] >= 10) & (df['Fare'] < 30)
    df['Fare_3'] = (df['Fare'] >= 30) & (df['Fare'] < 50)
    df['Fare_4'] = (df['Fare'] >= 50)
    df = df.drop(columns='Fare')


    print('33333',df.head())
    df['Name']=df['Name'].str.split(', ').str[1].str.split('. ').str[0]
    df['Master']=(df['Name']=='Master')
    df['Mr']=(df['Name']=='Mr')
    df['Miss']=(df['Name']=='Miss')
    df['Mrs']=(df['Name']=='Mrs')

    df = df.drop(columns = 'Name')


  

    #dfframe 행
    cols = ['Pclass1','Pclass2','Pclass3','Sex','Age1','Age2','Age3','Age4','Age5','Age6','Em_C','Em_S','Em_Q', 'Master','Mr','Miss','Mrs','1_3','only','many','Fare_1','Fare_2','Fare_3','Fare_4']

    for i in cols:
        df = df.astype({i:int})
    return df


if __name__ == '__main__':
    seed = 3
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # CSV파일 읽어오기
    df_train = pd.read_csv('./train.csv')
    df_test = pd.read_csv('./test.csv')
    gender = pd.read_csv('./gender_submission.csv')


    df_train = non_df(df_train)
    df_test = non_df(df_test)


    df_train = df_normalization(df_train)
    df_test = df_normalization(df_test)

    train_val = df_train.values
    test_val = df_test.values
    X = train_val[:,1:25].astype(float)
    Y = train_val[:,0].astype(float)

    X_test = test_val[:,1:25].astype(float)
    Y_test = test_val[:,0].astype(float)

    n_fold = 10
    # 교차 검증하기 (sklearn)
    skf = StratifiedKFold(n_splits = n_fold, shuffle = True, random_state = seed)
    accuracy2 = []
    i = 0
    for train, val in skf.split(X,Y):
        model = Sequential()
        # input_dim = dfframe 행개수
        model.add(Dense(1, input_dim = 24, activation = 'relu', kernel_initializer ='uniform'))
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
        gender['Survived'] = predictions
        i = i+1 
        gender.to_csv(f'./kim_submission{i}.csv',index=False) #csv 파일저장

print(accuracy2)    
