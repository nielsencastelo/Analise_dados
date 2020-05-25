# -*- coding: utf-8 -*-
"""
Created on Fri May 22 10:39:25 2020

@author: Nielsen
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.decomposition import PCA

base = pd.read_csv('dataset_test_ds.csv', sep = ';')

safra = base[['Safra']]
target = base[['TARGET']]

X = base[['V1','V2','V3', 'V4','V5','V6','V7','V8','V9','V10']]

# GETTING Correllation matrix
corr_mat = X.corr(method='pearson')
plt.figure(figsize=(10,10))
sns.heatmap(corr_mat,vmax=1,square=True,annot=True,cmap='cubehelix')

base.hist(figsize=(8,16), layout=(6,2), alpha=0.5);

X_array = np.asarray(X)


# Finding normalised array of X_Train
X_std = StandardScaler().fit_transform(X)

pca = PCA().fit(X_std)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlim(0,7,1)
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')



# Since 5 components can explain more than 70% of the variance, 
# we choose the number of the components to be 5
sklearn_pca = PCA(n_components=5)
X_Train = sklearn_pca.fit_transform(X_std)

sns.set(style='darkgrid')
f, ax = plt.subplots(figsize=(8, 8))
# ax.set_aspect('equal')
ax = sns.kdeplot(X_Train[:,0], X_Train[:,1], cmap="Greens",
          shade=True, shade_lowest=False)
ax = sns.kdeplot(X_Train[:,1], X_Train[:,2], cmap="Reds",
          shade=True, shade_lowest=False)
ax = sns.kdeplot(X_Train[:,2], X_Train[:,3], cmap="Blues",
          shade=True, shade_lowest=False)
red = sns.color_palette("Reds")[-2]
blue = sns.color_palette("Blues")[-2]
green = sns.color_palette("Greens")[-2]
ax.text(0.5, 0.5, "2nd and 3rd Projection", size=12, color=blue)
ax.text(-4, 0.0, "1st and 3rd Projection", size=12, color=red)
ax.text(2, 0, "1st and 2nd Projection", size=12, color=green)
plt.xlim(-6,5)
plt.ylim(-2,2)

number_of_samples = len(X_Train)
np.random.seed(0)

random_indices = np.random.permutation(number_of_samples)
num_training_samples = int(number_of_samples*0.75)

x_train = X_Train[random_indices[:num_training_samples]]

y_train = y[random_indices[:num_training_samples]]

x_test = X_Train[random_indices[num_training_samples:]]
y_test = y[random_indices[num_training_samples:]]
y_Train = list(y_train)

#Ridge Regression
model = linear_model.Ridge()
model.fit(x_train,y_train)
y_predict = model.predict(x_train)  

error=0
for i in range(len(y_Train)):
    error+=(abs(y_Train[i]-y_predict[i])/y_Train[i])
train_error_ridge=error/len(y_Train)*100
print("Train error = "'{}'.format(train_error_ridge)+" percent in Ridge Regression")

Y_test=model.predict(x_test)
y_Predict=list(y_test)

error=0
for i in range(len(y_test)):
    error+=(abs(y_Predict[i]-Y_test[i])/y_Predict[i])
test_error_ridge=error/len(Y_test)*100
print("Test error = "'{}'.format(test_error_ridge)+" percent in Ridge Regression")