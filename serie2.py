# -*- coding: utf-8 -*-
"""
Created on Sun May 24 21:58:00 2020

@author: Nielsen
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv('dataset_test_ds.csv', sep = ';')
X = df.drop(columns=['Safra'])