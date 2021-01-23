# -*- coding: utf-8 -*-
"""
Created on Mon May 25 10:22:19 2020

@author: Nielsen
"""

from pandas import DataFrame

MAE = [4.87, 2.89, 2.93, 2.90, 4.26, 0.30]
MSE = [53.67, 23.34, 22.69, 23.01, 74.90, 28.79]
RMSE= [2.20, 1.70, 1.71, 1.70, 2.06, 1.73]
sc  = [0.31, 0.70, 0.71, 0.70, 0.43, 0.63]

col={'MAE':MAE,'MSE':MSE,'RMSE': RMSE, 'Score': sc}
models=['Decision Tree','Bayesiano','Random Forest','Polinomial','Logística','SVM']

df=DataFrame(data=col,index=models)
df.plot(kind='bar')