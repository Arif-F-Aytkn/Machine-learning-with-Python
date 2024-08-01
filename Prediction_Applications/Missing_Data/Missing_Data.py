# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk

data = pd.read_csv("C:\\Users\\Arif Furkan\\OneDrive\\Belgeler\\Python_kullanirken\\eksikveriler.csv")
print(data)

# Filling Missing Data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
Age = data.iloc[:, 1:4].values  
print(Age)
imputer = imputer.fit(Age[:, 1:4]) 
Age[:, 1:4] = imputer.transform(Age[:, 1:4])
print(Age)
Country = data.iloc[:, 0:1].values
print(Country)

# Encoding Categorical Data
from sklearn import preprocessing
le = preprocessing.LabelEncoder() 
Country[:, 0] = le.fit_transform(data.iloc[:, 0])
print(Country)
ohe = preprocessing.OneHotEncoder()  
Country = ohe.fit_transform(Country).toarray() 
print(Country)

# Creating DataFrames
print(list(range(22)))
result = pd.DataFrame(data=Country, index=range(22), columns=['fr', 'tr', 'us'])
print(result)
result2 = pd.DataFrame(data=Age, index=range(22), columns=['Size', 'Weight', 'Age'])
print(result2)
sex = data.iloc[:, -1].values
print(sex)
result3 = pd.DataFrame(data=sex, index=range(22), columns=['sex'])
print(result3) 

# Merging All Data
r = pd.concat([result, result2], axis=1) 
print(r)
r2 = pd.concat([r, result3], axis=1)
print(r2) 

# Splitting Data into Training and Testing Sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(r, result3, test_size=0.33, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler() 
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test) 
y_train = np.ravel(y_train.to_numpy())
y_test = np.ravel(y_test.to_numpy())
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0) 
logr.fit(X_train, y_train)
y_pred = logr.predict(X_test)
print(y_pred)
print(y_test)