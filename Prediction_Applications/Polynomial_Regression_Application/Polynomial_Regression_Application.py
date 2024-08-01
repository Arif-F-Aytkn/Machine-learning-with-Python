# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk

data = pd.DataFrame(pd.read_csv("C:\\Users\\Arif Furkan\\OneDrive\\Belgeler\\Python_kullanirken\\maaslar.csv"))
print(data)

# Prepare Data
x = data.iloc[:, 1:2]
y = data.iloc[:, 2:]
X = x.values
Y = y.values

# Simple Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)

plt.scatter(X, Y)
plt.plot(x, lin_reg.predict(X))
plt.show()

# Polynomial Regression (Degree=2)
from sklearn.preprocessing import PolynomialFeatures
poly_reg2 = PolynomialFeatures(degree=2)
x_poly2 = poly_reg2.fit_transform(X)
print(x_poly2)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly2, y)

plt.scatter(X, Y, color="red")
plt.plot(X, lin_reg2.predict(poly_reg2.transform(X)))
plt.show()

# Polynomial Regression (Degree=4)
poly_reg4 = PolynomialFeatures(degree=4)
x_poly4 = poly_reg4.fit_transform(X)
lin_reg4 = LinearRegression()
lin_reg4.fit(x_poly4, y)

plt.scatter(X, Y, color="red")
plt.plot(X, lin_reg4.predict(poly_reg4.transform(X)))
plt.show()

# Predictions
print("Linear Regression Prediction (Position Level=11):", lin_reg.predict([[11]]))
print("Linear Regression Prediction (Position Level=6.6):", lin_reg.predict([[6.6]]))

print("Polynomial Regression Prediction (Degree=2, Position Level=6.6):", lin_reg2.predict(poly_reg2.transform([[6.6]])))
print("Polynomial Regression Prediction (Degree=2, Position Level=11):", lin_reg2.predict(poly_reg2.transform([[11]])))
