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
def non_data(df):
    df['Age'].fillna(df['Age'].mean(), inplace = True) # 나이 결측치 평균으로 맞추기
    df['Cabin'].fillna(df['Cabin'].mean(), inplace = True) # Cabin 결측치 평균으로 맞추기
    df['Embarked'].fillna(random.choice(embarked), inplace = True) # 랜덤으로 Embarked 정해주기
    df = df.drop(['Ticket','Cabin']) # 필요없는 데이터 삭제

    return df

def data_normalization(df):
    df = df.set_index("PassengerId")
    df.loc[df['Sex'] =='male'] = 0 # 남자 0으로 변환
    df.loc[df['Sex'] =='female'] = 1 # 여자 1로 변환

    # Pclass 별로 나누기
    df['Pclass1'] = (df['Pclass'] == 1)
    df['Pclass2'] = (df['Pclass'] == 2) 
    df['Pclass3'] = (df['Pclass'] == 3)
    df.drop(columns='Pclass') # pclass 행 삭제

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
    df['Name'] = df['Name'].str.split(',').str[1].str.split('.').str[0]
    df['Master'] = df['Name'] == 'Master'
    df['Mr'] = df['Name'] == 'Mr'
    df['Miss'] = df['Name'] == 'Miss'
    df['Mrs'] = df['Name'] == 'Mrs'

    df = df.drop(columns='Embarked') # Embarked 행 삭제하기

    
