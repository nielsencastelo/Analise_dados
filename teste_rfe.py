# -*- coding: utf-8 -*-
"""
Created on Sun May 24 16:24:27 2020

@author: Nielsen
"""


from sklearn.feature_selection import RFE
from sklearn.svm import SVR
import pandas as pd

base = pd.read_csv('dataset_test_ds.csv', sep = ';')

X = base.copy()
y = base.TARGET

X = base.drop(columns=['Safra','TARGET'])

estimator = SVR(kernel="linear")
selector = RFE(estimator, n_features_to_select=5, step=1)
selector = selector.fit(X, y)
selector.support_


selector.ranking_