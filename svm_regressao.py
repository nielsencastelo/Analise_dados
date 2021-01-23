# -*- coding: utf-8 -*-
"""
Created on Sun May 24 12:19:35 2020

@author: Nielsen
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import mean_squared_error,  mean_absolute_error
import numpy as np


# Preparativos iniciais
file = 'listings.csv'

df = pd.read_csv(file, sep = ',')

lista_colunas = df.columns


#Selecao dos atributos que serao usados para reviews_score_rating
base = df[['review_scores_rating','number_of_reviews','number_of_reviews_ltm',
           'number_of_reviews_l30d','review_scores_accuracy','review_scores_cleanliness',
           'review_scores_checkin','review_scores_communication','review_scores_location',
           'review_scores_value','reviews_per_month']]

# Tratamento de Dados
base = base.dropna()

base['review_scores_rating'] = base['review_scores_rating'].astype(int)
base['number_of_reviews'] = base['number_of_reviews'].astype(int)
base['number_of_reviews_ltm'] = base['number_of_reviews_ltm'].astype(int)
base['number_of_reviews_l30d'] = base['number_of_reviews_l30d'].astype(int)
base['review_scores_accuracy'] = base['review_scores_accuracy'].astype(int)
base['review_scores_cleanliness'] = base['review_scores_cleanliness'].astype(int)
base['review_scores_checkin'] = base['review_scores_checkin'].astype(int)
base['review_scores_communication'] = base['review_scores_communication'].astype(int)
base['review_scores_location'] = base['review_scores_location'].astype(int)
base['review_scores_value'] = base['review_scores_value'].astype(int)
base['reviews_per_month'] = base['reviews_per_month'].astype(float)

X = base.copy()
y = base.review_scores_rating

X = base.drop(columns=['review_scores_rating'])


# separa os dados de teste e treinamento com 70% de treinamento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    random_state=42)
# Cria o modelo
svm_reg=svm.SVR()
svm_reg.fit(X_train,y_train)
pred = svm_reg.predict(X_test)


print('MAE:', mean_absolute_error(pred, y_test))
print('MSE:', mean_squared_error(pred, y_test))
print('RMSE:', np.sqrt(mean_absolute_error(pred, y_test)))
print('Score:', svm_reg.score(X_test, y_test))