# -*- coding: utf-8 -*-
"""
Created on Mon May 25 10:22:19 2020

@author: Nielsen
"""

from pandas import DataFrame

MAE = [0.0237, 0.0236, 0.0235, 0.0193, 0.0216, 0.0116, 0.2056]
MSE = [0.0102, 0.0102, 0.0103, 0.0101, 0.0100, 0.0116, 1.0215]
RMSE= [0.1539, 0.1539, 0.1533, 0.1390, 0.1471, 0.1078, 0.4534]
r2  = [0.0706, 0.0705, 0.0687, 0.0824, 0.0915, 0.9883, 0.0187]

col={'MAE':MAE,'MSE':MSE,'RMSE': RMSE, 'r2': r2}
models=['Regressão','Ridge','Bayesiano','Random Forest','Polinomial','Logística','SVM']

df=DataFrame(data=col,index=models)
df.plot(kind='bar')