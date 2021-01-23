# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 20:18:01 2021

@author: niels
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# regressao e previsoes

def remover(valor):
    
    return valor.replace('%','')

def remover2(valor):
    
    return valor.replace('$','').replace(',','')

file = 'listings.csv'


df = pd.read_csv(file, sep = ',')

lista_colunas = df.columns

# selecao de atributos sem analise exploratoria
base = df[['host_location', 'host_acceptance_rate','price', 'property_type', 
           'room_type','accommodates', 'bedrooms' ,  'beds', 'amenities',
           'minimum_nights', 'maximum_nights', 'instant_bookable']]

# pre processamento
base = base.dropna()

base['host_acceptance_rate'] = base['host_acceptance_rate'].apply(remover)
base['price']  = base['price'].apply(remover2)
base['price'] = base['price'].astype(float)
base['host_acceptance_rate'] = base['host_acceptance_rate'].astype(int)
base['instant_bookable'] = base['instant_bookable'].astype(bool)
base['bedrooms'] = base['bedrooms'].astype(int)
base['beds'] = base['beds'].astype(int)

lista_colunas = base.columns

# matriz de correlacao

df_matriz = base[['host_acceptance_rate', 'price', 'accommodates', 'bedrooms', 
                  'beds', 'minimum_nights', 'maximum_nights']]

mat_corr = df_matriz.corr(method='pearson')

# GETTING Correllation matrix
#corr_mat = X.corr(method='pearson')
plt.figure(figsize=(10,10))
sns.heatmap(mat_corr,vmax=1,square=True,annot=True,cmap='cubehelix')

base.hist(figsize=(8,16), layout=(6,2), alpha=0.5);