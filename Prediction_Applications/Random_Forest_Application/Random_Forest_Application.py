# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
data = pd.read_csvpd.read_csv("C:\\Users\\Arif Furkan\\OneDrive\\Belgeler\\Python_kullanirken\\maaslar.csv")
print(data)

# Prepare the feature and target variables
x = data.iloc[:, 1:2]
y = data.iloc[:, 2:]
X = x.values
Y = y.values

# Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)
plt.scatter(X, Y, color='red')
plt.plot(x, lin_reg.predict(X), color='blue')
plt.show()

# Polynomial Regression (Degree = 2)
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)
plt.scatter(X, Y, color='red')
plt.plot(X, lin_reg2.predict(poly_reg.transform(X)), color='blue')
plt.show()

# Polynomial Regression (Degree = 4)
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)
plt.scatter(X, Y, color='red')
plt.plot(X, lin_reg2.predict(poly_reg.transform(X)), color='blue')
plt.show()

# Predictions
print("Linear Regression Prediction (Position Level=11):", lin_reg.predict([[11]]))
print("Linear Regression Prediction (Position Level=6.6):", lin_reg.predict([[6.6]]))
print("Polynomial Regression Prediction (Degree=2, Position Level=6.6):", lin_reg2.predict(poly_reg.transform([[6.6]])))
print("Polynomial Regression Prediction (Degree=2, Position Level=11):", lin_reg2.predict(poly_reg.transform([[11]])))

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
x_scaled = sc1.fit_transform(X)
sc2 = StandardScaler()
y_scaled = np.ravel(sc2.fit_transform(Y.reshape(-1, 1)))

# Support Vector Regression (SVR)
from sklearn.svm import SVR
svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_scaled, y_scaled)
plt.scatter(x_scaled, y_scaled, color='red')
plt.plot(x_scaled, svr_reg.predict(x_scaled), color='blue')
plt.show()
print("SVR Prediction (Position Level=11):", svr_reg.predict(sc1.transform([[11]])))
print("SVR Prediction (Position Level=6.6):", svr_reg.predict(sc1.transform([[6.6]])))

# Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X, Y)
Z = X + 0.5
K = X - 0.4
plt.scatter(X, Y, color='red')
plt.plot(x, r_dt.predict(X), color='blue')
plt.plot(x, r_dt.predict(Z), color='green')
plt.plot(x, r_dt.predict(K), color='yellow')
plt.show()
print("Decision Tree Prediction (Position Level=11):", r_dt.predict([[11]]))
print("Decision Tree Prediction (Position Level=6.6):", r_dt.predict([[6.6]]))

# Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=10, random_state=0)
rf_reg.fit(X, Y.ravel())
print("Random Forest Prediction (Position Level=6.5):", rf_reg.predict([[6.5]]))
plt.scatter(X, Y, color='red')
plt.plot(X, rf_reg.predict(X), color='blue')
plt.plot(X, rf_reg.predict(Z), color='green')
plt.plot(x, r_dt.predict(K), color='yellow')
plt.show()