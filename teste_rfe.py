# -*- coding: utf-8 -*-
"""
Created on Sun May 24 16:24:27 2020

@author: Nielsen
"""


from sklearn.feature_selection import RFE
from sklearn.svm import SVR
import pandas as pd

file = 'listings.csv'

df = pd.read_csv(file, sep = ',')



base = df[['review_scores_rating','number_of_reviews','number_of_reviews_ltm',
           'number_of_reviews_l30d','review_scores_accuracy','review_scores_cleanliness',
           'review_scores_checkin','review_scores_communication','review_scores_location',
           'review_scores_value','reviews_per_month']]

lista_colunas = base.columns

base = base.dropna()

X = base.copy()
y = base.review_scores_rating

X = base.drop(columns=['review_scores_rating'])

estimator = SVR(kernel="linear")
selector = RFE(estimator, n_features_to_select=5, step=1)
selector = selector.fit(X, y)
selector.support_


selector.ranking_