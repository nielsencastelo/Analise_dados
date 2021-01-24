# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 09:47:49 2021

@author: niels
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,  mean_absolute_error
from sklearn.model_selection import cross_val_score


from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

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
y = base['review_scores_rating'].values

X = base.drop(columns=['review_scores_rating'])


def criar_rede():
    model = Sequential()
    model.add(Dense(units = 158, activation = 'relu', input_dim = 10))
    model.add(Dense(units = 158, activation = 'relu'))
    model.add(Dense(units = 1, activation = 'linear'))
    model.compile(loss = 'mean_absolute_error', optimizer = 'adam',
                      metrics = ['mean_absolute_error'])
    return model

regressor = KerasRegressor(build_fn = criar_rede,
                           epochs = 100,
                           batch_size = 256)

resultados = cross_val_score(estimator = regressor,
                             X = X, y = y,
                             cv = 10)

media = resultados.mean()
desvio = resultados.std()