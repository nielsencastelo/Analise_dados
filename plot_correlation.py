import pandas as pd
import numpy as np
from main import *
import seaborn as sns

X = pd.read_csv("dataset_test_ds.csv",sep=';')

for index, X in X.groupby(['Safra']):
    y = X.TARGET
    X.drop(columns=['TARGET','Safra'],inplace=True)

    corr = X.corr()
    corr.style.background_gradient(cmap='coolwarm')

    sns.heatmap(corr, annot=True, fmt=".2f")

    plt.title(str(index))
    plt.suptitle("Bar Chart")

    plt.savefig(str(index)+'.png', bbox_inches='tight')

    plt.clf()
