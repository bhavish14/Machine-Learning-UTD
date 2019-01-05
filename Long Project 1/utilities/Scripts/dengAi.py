from __future__ import print_function
from __future__ import division

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
#import statsmodels.api as sm


train_features = pd.read_csv(
    '/Users/bhavish96.n/Documents/UTD/Fall \'18/Machine Learning [Anurag Nagar]/Assignments/Long Project 1/Project Data/dengue_features_train.csv',
    index_col = [0, 1, 2]
)
train_lables = pd.read_csv(
    '/Users/bhavish96.n/Documents/UTD/Fall \'18/Machine Learning [Anurag Nagar]/Assignments/Long Project 1/Project Data/dengue_labels_train.csv',
    index_col = [0, 1, 2]
)


sj_train_features = train_features.loc['sj']
sj_train_labels = train_lables.loc['sj']

iq_train_features = train_features.loc['iq']
iq_train_labels = train_lables.loc['iq']

print('San Juan')
print('features: ', sj_train_features.shape)
print('labels  : ', sj_train_labels.shape)

print('\nIquitos')
print('features: ', iq_train_features.shape)
print('labels  : ', iq_train_labels.shape)

print (pd.isnull(sj_train_features).any())

# Remove `week_start_date` string.
sj_train_features.drop('week_start_date', axis=1, inplace=True)
iq_train_features.drop('week_start_date', axis=1, inplace=True)

sj_train_features.fillna(method = 'ffill', inplace = True)
iq_train_features.fillna(method = 'ffill', inplace = True)

print (pd.isnull(sj_train_features).any())

# poission -> mean and variance is equal
# negative binomial regression -> no assumptions as such

print('San Juan')
print('mean: ', sj_train_labels.mean()[0])
print('var :', sj_train_labels.var()[0])

print('\nIquitos')
print('mean: ', iq_train_labels.mean()[0])
print('var :', iq_train_labels.var()[0])

# principle component analysis
sj_train_features['total_cases'] = sj_train_labels.total_cases
iq_train_features['total_cases'] = iq_train_labels.total_cases

sj_corr = sj_train_features.corr()
iq_corr = iq_train_features.corr()

# plot san juan
sj_corr_heat = sns.heatmap(sj_corr)
plt.title('San Juan Variable Correlations')
plt.show()

(sj_corr
     .total_cases
     .drop('total_cases') # don't compare with myself
     .sort_values(ascending=False)
     .plot
     .barh())

plt.show()

# plot iquitos
iq_corr_heat = sns.heatmap(iq_corr)
plt.title('Iquitos Variable Correlations')
plt.show()

(iq_corr
     .total_cases
     .drop('total_cases') # don't compare with myself
     .sort_values(ascending=False)
     .plot
     .barh())

plt.show()