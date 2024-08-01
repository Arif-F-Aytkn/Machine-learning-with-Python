# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk

data = pd.read_csv("C:\\Users\\Arif Furkan\\OneDrive\\Belgeler\\Python_kullanirken\\satislar.csv")
print(data)
months = data[["Aylar"]]
sales = data[["Satislar"]]
sales2 = data.iloc[:,:1].values

# Split the Dataset into Training and Testing Sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(months, sales, test_size=0.33, random_state=0)

# Scaling the data
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()

x_train_scaled = sc_x.fit_transform(x_train)
x_test_scaled = sc_x.transform(x_test)

y_train_scaled = sc_y.fit_transform(y_train)
y_test_scaled = sc_y.transform(y_test)

"""
print(x_train)
print(x_test)
print(y_train)
print(y_test)
"""

# Train the Linear Regression Model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train_scaled, y_train_scaled)

# Make Predictions
prediction = sc_y.inverse_transform(prediction_scaled)
x_train_sorted = x_train.sort_values(by="Aylar")
y_train_sorted = y_train.loc[x_train_sorted.index]
x_test_sorted = x_test.sort_values(by="Aylar")
prediction_sorted = prediction[np.argsort(x_test.values.ravel())]

# Plot the Results
plt.plot(x_train_sorted, y_train_sorted, label="Training Data", color="blue", marker='o')
plt.plot(x_test_sorted, prediction_sorted, label="Predictions", color="red", marker='x')
plt.title("Sales by Months")
plt.xlabel("Months")
plt.ylabel("Sales")
plt.legend()
plt.show()