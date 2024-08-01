# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk

data = pd.DataFrame(pd.read_csv("C:\\Users\\Arif Furkan\\OneDrive\\Belgeler\\Python_kullanirken\\veriler.csv"))
print(data)

# Encoding Categorical Data for Country
country = data.iloc[:, 0:1].values
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
country[:, 0] = le.fit_transform(data.iloc[:, 0])
ohe = preprocessing.OneHotEncoder()
country = ohe.fit_transform(country).toarray()

# Encoding Categorical Data for Gender
from sklearn import preprocessing
gender = data.iloc[:, -1]
le = preprocessing.LabelEncoder()
data.iloc[:, -1] = le.fit_transform(gender)

# Creating DataFrames
print(list(range(22)))
result = pd.DataFrame(data=country, index=range(22), columns=['fr', 'tr', 'us'])
age = data.iloc[:, 1:4].values 
result2 = pd.DataFrame(data=age, index=range(22), columns=['height', 'weight', 'age'])
gender = data.iloc[:, -1].values
result3 = pd.DataFrame(data=gender, index=range(22), columns=['gender']) 

s = pd.concat([result, result2], axis=1) 
print(s)
s2 = pd.concat([s, result3], axis=1)
print(s2) 

# Splitting Data into Training and Test Sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(s, result3, test_size=0.33, random_state=0)

# Feature Scaling
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)

# Separating the Height Column
height = s2.iloc[:, 3:4].values
print(height)

# Separating Features and Splitting Data
left = s2.iloc[:, :3]
right = s2.iloc[:, 4:]
data = pd.concat([left, right], axis=1)
x_train, x_test, y_train, y_test = train_test_split(data, height, test_size=0.33, random_state=0)

# Training and Prediction with Regression Model
r2 = LinearRegression()
r2.fit(x_train, y_train)
y_pred = r2.predict(x_test)

# Creating Statistical Model
# Model with All Features
import statsmodels.api as sm
X = np.append(arr=np.ones((22, 1)).astype(int), values=data, axis=1)
X_l = data.iloc[:, [0, 1, 2, 3, 4, 5]].values
X_l = np.array(X_l, dtype=float)
model = sm.OLS(height, X_l).fit()
print(model.summary())

# Removing One Feature and Updating Model
X_l = data.iloc[:, [0, 1, 2, 3, 5]].values
X_l = np.array(X_l, dtype=float)
model = sm.OLS(height, X_l).fit()
print(model.summary())

# Removing Another Feature and Updating Model
X_l = data.iloc[:, [0, 1, 2, 3]].values
X_l = np.array(X_l, dtype=float)
model = sm.OLS(height, X_l).fit()
print(model.summary())
