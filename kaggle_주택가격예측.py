import pandas as pd
import random

df = pd.read_csv("./test.csv")
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

df['OverallQual1'] = (df['OverallQual']>=1) & (df['OverallQual'] <= 5)
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
df
