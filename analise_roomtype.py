# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 11:06:22 2021
Room Type
@author: Tom
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Preparativos iniciais

file = 'listings.csv'

df = pd.read_csv(file, sep = ',')

lista_colunas = df.columns


#Selecao dos atributos que serao usados para room type

base = df[['accommodates','bedrooms','amenities','property_type','room_type']]



#removi neighbourhood do fluxo pois não mostrava informações pertinentes

# Tratamento de Dados
base = base.dropna()

base['accommodates'] = base['accommodates'].astype(int)
base['bedrooms'] = base['bedrooms'].astype(int)

lista_colunas = base.columns

labelencoder_amenities = LabelEncoder()
labelencoder_property_type = LabelEncoder()
labelencoder_room_type = LabelEncoder()


base['amenities'] = labelencoder_amenities.fit_transform(base['amenities'].values)
base['property_type'] = labelencoder_property_type.fit_transform(base['property_type'].values)
base['room_type'] = labelencoder_room_type.fit_transform(base['room_type'].values)

mat_corr = base.corr(method='pearson')

# GETTING Correllation matrix
#corr_mat = X.corr(method='pearson')
plt.figure(figsize=(10,10))
sns.heatmap(mat_corr,vmax=1,square=True,annot=True,cmap='cubehelix')

base.hist(figsize=(8,16), layout=(6,2), alpha=0.5);