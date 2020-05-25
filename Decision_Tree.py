# -*- coding: utf-8 -*-
"""
Created on Sun May 24 12:17:25 2020

@author: Nielsen
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import mean_squared_error,  mean_absolute_error
import numpy as np

base = pd.read_csv('dataset_test_ds.csv', sep = ';')

X = base[['V1','V2','V3', 'V4','V5','V6','V7','V8','V9','V10']]
target = base[['TARGET']]


# separa os dados de teste e treinamento com 70% de treinamento
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, 
                                                    random_state=42) 
# Cria o modelo
dec = tree.DecisionTreeRegressor(max_depth=1)
dec.fit(X_train,y_train)
pred = dec.predict(X_test)


print('MAE:', mean_absolute_error(pred, y_test))
print('MSE:', mean_squared_error(pred, y_test))
print('RMSE:', np.sqrt(mean_absolute_error(pred, y_test)))
print('Score:', dec.score(X_test, y_test))

