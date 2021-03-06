# -*- coding: utf-8 -*-
"""Cancer Dataset MLP.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MF8QaLB5Y9mBXB1YqtNAI_5URgD1KfVT
"""

#Data Preprocessing


from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

cancer_dataset = load_breast_cancer()
cancer_dataset.keys()
cancer_dataset
cancer_dataset['data'].shape
df = pd.DataFrame(cancer_dataset.data, columns=cancer_dataset.feature_names)
df.head
df.iloc[:, 5:8].hist()

#Correlations
correlartions = df.iloc[:, 5:8].corr()
print (correlartions)

figure = plt.figure()
ax = figure.add_subplot(111)
cax = ax.matshow(correlartions, vmin = -1, vmax = 1)
figure.colorbar(cax)
ticks = np.arange(0, 3, 1)
ax.set_xticks(ticks)
ax.set_xticklabels(df.iloc[:, 5:8].columns)
ax.set_yticks(ticks)
ax.set_yticklabels(df.iloc[:, 5:8].columns)
plt.show()

scatter_matrix(df.iloc[:, 5:8])
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

X = cancer_dataset['data']
y = cancer_dataset['target']

X_train, X_test, y_train, y_test = train_test_split(X, y)

scaler = StandardScaler()
scaler.fit(X_train)

print (np.amin(X_train))
print (np.amax(X_train))
print (np.amin(X_test))
print (np.amax(X_test))

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


print (np.amin(X_train))
print (np.amax(X_train))
print (np.amin(X_test))
print (np.amax(X_test))

mlp = MLPClassifier(hidden_layer_sizes = (30, 30, 30))
mlp.fit(X_train, y_train)
predictions = mlp.predict(X_test)
print (confusion_matrix(y_test, predictions))
print (classification_report(y_test, predictions))