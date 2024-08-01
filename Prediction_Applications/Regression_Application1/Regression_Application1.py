# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Load the dataset
data = pd.read_csv("C:\\Users\\Arif Furkan\\OneDrive\\Belgeler\\Python_kullanirken\\odev_tenis.csv")
print(data)

# Encode categorical features
label_encoder = LabelEncoder()
data_encoded = data.apply(label_encoder.fit_transform)

# One-hot encode the weather data
column_transformer = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [0])],
    remainder='passthrough')
weather_encoded = column_transformer.fit_transform(data_encoded.iloc[:, :1])

# Convert encoded features to DataFrame
weather_df = pd.DataFrame(data=weather_encoded, index=range(len(data)), columns=['o', 'r', 's'])

# Combine the one-hot encoded weather data with other features
processed_data = pd.concat([weather_df, data_encoded.iloc[:, 1:3]], axis=1)
processed_data = pd.concat([data_encoded.iloc[:, -2:], processed_data], axis=1)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(processed_data.iloc[:, :-1], processed_data.iloc[:, -1:], test_size=0.33, random_state=0)

# Train a Linear Regression model
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
print("Predictions:\n", y_pred)

# Backward elimination for feature selection
# Add a column of ones to the dataset for the intercept term
X = np.append(arr=np.ones((len(processed_data), 1)).astype(int), values=processed_data.iloc[:, :-1], axis=1)

# Initial model
X_1 = processed_data.iloc[:, [0, 1, 2, 3, 4, 5]].values
X_1 = np.array(X_1, dtype=float)
model = sm.OLS(processed_data.iloc[:, -1], X_1).fit()
print("Initial Model Summary:\n", model.summary())

# Remove one feature and update model
processed_data = processed_data.iloc[:, 1:]
X = np.append(arr=np.ones((len(processed_data), 1)).astype(int), values=processed_data.iloc[:, :-1], axis=1)
X_1 = processed_data.iloc[:, [0, 1, 2, 3, 4]].values
X_1 = np.array(X_1, dtype=float)
model = sm.OLS(processed_data.iloc[:, -1], X_1).fit()
print("Updated Model Summary (after feature removal):\n", model.summary())

# Train the model again after feature selection
x_train = x_train.iloc[:, 1:]
x_test = x_test.iloc[:, 1:]
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
print("Predictions after feature selection:\n", y_pred)
