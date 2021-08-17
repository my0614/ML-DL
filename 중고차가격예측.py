import pandas as pd
df = pd.read_csv('./test-data.csv')
df['Name'] = df['Name'].str.split(' ').str[0]

fuel = pd.get_dummies(df['Fuel_Type'])
name = pd.get_dummies(df['Name'])
location = pd.get_dummies(df['Location'])
owner = pd.get_dummies(df['Owner_Type'])
"""
df['Kilometers_Driven1'] = (df['Kilometers_Driven'] >= 0) & (df['Kilometers_Driven']  <= 50000)
df['Kilometers_Driven2'] = (df['Kilometers_Driven'] >= 50000) & (df['Kilometers_Driven']  <= 100000)
df['Kilometers_Driven3'] = (df['Kilometers_Driven'] >= 100000) & (df['Kilometers_Driven']  <= 200000)
df['Kilometers_Driven4'] = (df['Kilometers_Driven'] >= 200000) & (df['Kilometers_Driven']  <= 300000)
df['Kilometers_Driven5'] = (df['Kilometers_Driven'] >= 350000) 
"""

re = pd.concat([name,location,fuel,owner],axis = 1)
re
