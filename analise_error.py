# -*- coding: utf-8 -*-
"""
Created on Mon May 25 10:22:19 2020

@author: Nielsen
"""

from pandas import DataFrame

MAE = [4.87, 2.89, 2.93, 2.90, 4.26, 0.30, 0.38]
MSE = [53.67, 23.34, 22.69, 23.01, 74.90, 28.79, 0.15]
RMSE= [2.20, 1.70, 1.71, 1.70, 2.06, 1.73, 0.61]

col={'MAE':MAE,'MSE':MSE,'RMSE': RMSE}
models=['Decision Tree','Bayesiano','Random Forest','Polinomial','Logística','SVM', 'Neural']

df=DataFrame(data=col,index=models)
df.plot(kind='bar')