# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 13:32:53 2021

@author: Tom

Analise reviews_score_rating
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

#Criação da matriz 

df_matriz = base[['review_scores_rating','number_of_reviews','number_of_reviews_ltm',
                  'number_of_reviews_l30d','review_scores_accuracy',
                  'review_scores_cleanliness','review_scores_checkin',
                  'review_scores_communication','review_scores_location',
                  'review_scores_value','reviews_per_month']]

mat_corr = df_matriz.corr(method='pearson')

# GETTING Correllation matrix
#corr_mat = X.corr(method='pearson')
plt.figure(figsize=(10,10))
sns.heatmap(mat_corr,vmax=1,square=True,annot=True,cmap='cubehelix')

base.hist(figsize=(8,16), layout=(6,2), alpha=0.5);