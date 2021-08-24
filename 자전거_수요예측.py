import numpy as np
import pandas as pd
from sklearn import svm

test = pd.read_csv("./test.csv")
train = pd.read_csv("./train.csv")
sub = pd.read_csv("./submission.csv")

test['hour1'] = (test['hour'] >=0) & (test['hour'] <10)
test['hour2'] = (test['hour'] >=10) & (test['hour'] <=20)
del test['hour']

rain = pd.get_dummies(test['hour_bef_precipitation'])
pd.concat([test,rain],axis = 1)

test['hour_bef_windspeed'] =  test['hour_bef_windspeed'] >= 2.5
test['hour_bef_humidity'] = test['hour_bef_humidity'] >=60
del test['hour_bef_visibility']
del test['hour_bef_ozone']
del test['hour_bef_pm10']
test
test['late dust0'] =  (test['hour_bef_pm2.5'] >=0) &  (test['hour_bef_pm2.5'] <=30) # 좋음
test['late dust1'] =  (test['hour_bef_pm2.5'] >=31) &  (test['hour_bef_pm2.5'] <=80) # 보통
test['late dust2'] =  (test['hour_bef_pm2.5'] >=81) &  (test['hour_bef_pm2.5'] <=150) # 나쁨
test['late dust3'] =  (test['hour_bef_pm2.5'] >=151)# 매우 나쁨
del test['hour_bef_pm2.5']
test
