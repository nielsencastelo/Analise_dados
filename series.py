# -*- coding: utf-8 -*-
"""
Created on Sun May 24 19:29:09 2020

@author: Nielsen
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


base = pd.read_csv('dataset_test_ds.csv', sep = ';').set_index('Safra')


for index, data in base.groupby(['Safra']):
    print(index,'::',list(data.TARGET.value_counts()))

base_sort = base.sort_values(['Safra'])

#base_sort.index = pd.to_datetime(base_sort.index)
base_sort.head(10)

x = pd.DataFrame(index=range(0, len(base_sort)))
x['V2'] = base_sort['V2'].values


plt.figure(figsize=(15,5))
plt.plot(v2, label='V2')


