# -*- coding: utf-8 -*-
"""
Created on Sun May 24 12:25:07 2020

@author: Nielsen
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,  mean_absolute_error
import numpy as np

base = pd.read_csv('dataset_test_ds.csv', sep = ';')

X = base[['V1','V2','V3', 'V4','V5','V6','V7','V8','V9','V10']]
target = base[['TARGET']]


# separa os dados de teste e treinamento com 70% de treinamento
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, 
                                                    random_state=42) 

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Cria o modelo
regressor = LinearRegression()
regressor.fit(X_train_poly, y_train)
poly_test_pred = regressor.predict(X_test_poly)


print('MAE:', mean_absolute_error(poly_test_pred, y_test))
print('MSE:', mean_squared_error(poly_test_pred, y_test))
print('RMSE:', np.sqrt(mean_absolute_error(poly_test_pred, y_test)))
print('Score:', regressor.score(X_test_poly, y_test))
