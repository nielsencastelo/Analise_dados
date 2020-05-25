# -*- coding: utf-8 -*-
"""
Created on Sun May 24 12:48:55 2020

@author: Nielsen
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,  mean_absolute_error
import numpy as np
from sklearn.model_selection import GridSearchCV

base = pd.read_csv('dataset_test_ds.csv', sep = ';')

X = base[['V1','V2','V3', 'V4','V5','V6','V7','V8','V9','V10']]
target = base[['TARGET']]


# separa os dados de teste e treinamento com 70% de treinamento
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, 
                                                    random_state=42) 
# Cria o modelo
parametros = {'min_samples_leaf':[1,10], 'min_samples_split':[2,10], 'n_estimators': [100,250,500,750]}
modelo = RandomForestRegressor()

grid = GridSearchCV(modelo,parametros)
grid.fit(X_train,y_train.values.ravel())

modelo = grid.best_estimator_
pred = modelo.predict(X_test)


print('MAE:', mean_absolute_error(pred, y_test))
print('MSE:', mean_squared_error(pred, y_test))
print('RMSE:', np.sqrt(mean_absolute_error(pred, y_test)))
print('Score:', modelo.score(X_test, y_test))