# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.DataFrame(pd.read_csv("C:\\Users\\Arif Furkan\\OneDrive\\Belgeler\\Python_kullanirken\\maaslar.csv"))
print(data)
x = data.iloc[:,1:2]
y = data.iloc[:,2:]
X = x.values
Y = y.values


# Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)
plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg.predict(X), color = 'blue')
plt.show()


# Polynomial Regression (degree = 2)
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.show()

# Polynomial Regression (degree = 4)
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.show()

# Predictions
print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))
print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
x_scaled = sc1.fit_transform(X)
sc2 = StandardScaler()
y_scaled = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))


from sklearn.svm import SVR
svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_scaled, y_scaled)
plt.scatter(x_scaled, y_scaled, color='red')
plt.plot(x_scaled, svr_reg.predict(x_scaled), color='blue')
plt.show()
print(svr_reg.predict([[11]]))
print(svr_reg.predict([[6.6]]))


from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)
Z = X + 0.5
K = X - 0.4
plt.scatter(X, Y, color='red')
plt.plot(x, r_dt.predict(X), color='blue')
plt.plot(x, r_dt.predict(Z), color='green')
plt.plot(x, r_dt.predict(K), color='yellow')
plt.show()

print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))
