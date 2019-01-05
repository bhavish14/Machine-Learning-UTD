import numpy as np
import pandas as pd
import re

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
from string import ascii_letters

# Models
from sklearn.tree import DecisionTreeClassifier

# Evaluation Metrics
from sklearn.metrics import accuracy_score




class dengAi():
    def __init__(self, path_train, col_names_train, path_tfeatures, col_names_tfeatures):
        self.X_train = pd.read_csv(
            path_train, names = col_names_train, header = None, skiprows = 1
        )
        self.y_train = pd.read_csv(
            path_tfeatures, names = col_names_tfeatures, header = None, skiprows = 1
        )

        self.correlation_martix = 0
        self.train_fnames = col_names_train
        self.train_fnames.append("total_cases")
        # final train data
        self.labels = 0
        #= pd.DataFrame(self.train_predictions['total_cases'].values)
        #self.y_train.columns = ['total_cases']



    '''
        PreProcessing the data
    '''
    def preprocessing(self):
        '''
            Converting categorical data to numerical
        '''
        lec = LabelEncoder()

        columns = self.X_train.columns
        numerical_columns = self.X_train._get_numeric_data().columns
        categorical_columns = set(columns) - set(numerical_columns)

        for row_name in categorical_columns:
            self.X_train[row_name] = lec.fit_transform(self.X_train[row_name])

        # Appending X_train and y_train to remove null values
        self.X_train['total_cases'] = self.y_train['total_cases']

        '''
            Null Values
        '''
        # Null values count
        null_count = self.X_train.isna().sum()

        print ("########## Missing Data Statistics #########\n" + str(null_count))

        # Removing Null values
        self.X_train = self.X_train.dropna()

        '''
            Correlation Matrix
        '''
        # Computing correlation matrix
        self.correlation_martix = self.X_train.corr().abs()
        #self.correlation_martix_visualize()

        # Dropping highly correlated components
        # Select upper triangle of correlation matrix
        upper = self.correlation_martix.where(np.triu(np.ones(self.correlation_martix.shape), k=1).astype(np.bool))

        # Find index of feature columns with correlation greater than 0.8
        to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]

        for item in ['city', 'year', 'weekofyear', 'week_start_date', 'total_cases']:
            if item in to_drop:
                to_drop.remove(item)

        for item in to_drop:
            self.train_fnames.remove(item)

        # Dropping colums
        self.X_train = self.X_train.drop(to_drop, axis=1)

        '''
            Normalizing Data
        '''

        fields = [ item for item in self.train_fnames if item not in ['city', 'year', 'weekofyear', 'week_start_date', 'total_cases']]
        min_max = MinMaxScaler()
        for item in fields:
            x = self.X_train[fields].values
            x_scaled = min_max.fit_transform(x)
            self.X_train[item] = x_scaled

        self.y_train = self.X_train['total_cases']
        self.X_train = self.X_train.drop('total_cases', axis = 1)


    #def data_visualization(self):

    '''
        Training Models
    '''
    # Decision Tree
    def decision_tree(self):
        for item in ['city', 'year', 'weekofyear']:
            self.X_train = self.X_train.drop(item, axis = 1)
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_train, self.y_train, test_size = 0.33
        )
        d_clf = DecisionTreeClassifier()
        d_clf.fit(X_train, y_train)
        result = d_clf.predict(X_test)
        for item in zip(y_test, result):
            print (item)

        print(accuracy_score(result, y_test))



    # Utility functions

    '''
        Data Visualization
    '''
    def correlation_martix_visualize(self):
        sns.set(style = 'white')
        # Generate a mask for the upper triangle
        mask = np.zeros_like(self.correlation_martix, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(11, 9))

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(self.correlation_martix, mask=mask, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.show()


def main():

    path_train = "/Users/bhavish96.n/Documents/UTD/Fall '18/Machine Learning [Anurag Nagar]/Assignments/Long Project 1/Project Data/dengue_features_train.csv"
    train_names = ['city', 'year', 'weekofyear', 'week_start_date', 'ndvi_ne', 'ndvi_nw',
    'ndvi_se', 'ndvi_sw', 'precipitation_amt_mm', 'reanalysis_air_temp_k', 'reanalysis_avg_temp_k',
    'reanalysis_dew_point_temp_k', 'reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k',
    'reanalysis_precip_amt_kg_per_m2', 'reanalysis_relative_humidity_percent', 'reanalysis_sat_precip_amt_mm',
    'reanalysis_specific_humidity_g_per_kg', 'reanalysis_tdtr_k', 'station_avg_temp_c',
    'station_diur_temp_rng_c', 'station_max_temp_c', 'station_min_temp_c', 'station_precip_mm']

    path_tfeatures = "/Users/bhavish96.n/Documents/UTD/Fall '18/Machine Learning [Anurag Nagar]/Assignments/Long Project 1/Project Data/dengue_labels_train.csv"
    tfeatures_names = "city	year	weekofyear	total_cases".split()

    deng_object = dengAi(path_train, train_names, path_tfeatures, tfeatures_names)

    deng_object.preprocessing()
    deng_object.correlation_martix_visualize()
    deng_object.decision_tree()


if __name__ == '__main__':
    main()
