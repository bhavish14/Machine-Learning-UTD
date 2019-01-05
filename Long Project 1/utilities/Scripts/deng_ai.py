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
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Evaluation Metrics
from sklearn.metrics import accuracy_score

class dengAi():
    def __init__(self, train_path, train_names, train_results_path, train_result_names, path_test, path_test_results):
        self.X_train = pd.read_csv(
            train_path, names = train_names, header = None, skiprows = 1
        )
        self.y_train = pd.read_csv(
            train_results_path, names = train_result_names, header = None, skiprows = 1
        )

        '''
        self.X_test = pd.read_csv(
            path_test, names = train_names, header = None, skiprows = 1
        )
        '''

        self.correlation_martix = 0
        self.train_features = train_names


    def preprocess_data(self):
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

        # Null values count
        null_count = self.X_train.isna().sum()

        print ("########## Missing Data Statistics #########\n" + str(null_count))

        # Removing Null values
        self.X_train = self.X_train.dropna()

        '''
            Correlation Matrix
        '''


        '''
            Normalizing Data
        '''

    '''
        Training Models
    '''
    # Decision Tree
    def decision_tree(self):
        self.y_train = self.X_train['total_cases']
        self.X_train = self.X_train.drop(['city', 'week_start_date', 'year', 'weekofyear', 'total_cases'], axis = 1)
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_train, self.y_train, test_size = 0.33
        )
        d_clf = DecisionTreeRegressor()
        d_clf.fit(X_train, y_train)
        result = d_clf.predict(X_test)

        print(result)
        print(accuracy_score(result, y_test))


    def svm(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_train, self.y_train, test_size = 0.33
        )
        svm_clf = SVC()
        svm_clf.fit(X_train, y_train)
        result = svm_clf.predict(X_test)

        print(accuracy_score(result, y_test))

    def random_forest(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_train, self.y_train, test_size = 0.33
        )
        rf_clf = RandomForestClassifier()
        rf_clf.fit(X_train, y_train)
        result = rf_clf.predict(X_test)

        print(accuracy_score(result, y_test))



def main():

    path_train = "/Users/bhavish96.n/Documents/UTD/Fall '18/Machine Learning [Anurag Nagar]/Assignments/Long Project 1/Project Data/dengue_features_train.csv"
    train_names = ['city', 'year', 'weekofyear', 'week_start_date', 'ndvi_ne', 'ndvi_nw',
    'ndvi_se', 'ndvi_sw', 'precipitation_amt_mm', 'reanalysis_air_temp_k', 'reanalysis_avg_temp_k',
    'reanalysis_dew_point_temp_k', 'reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k',
    'reanalysis_precip_amt_kg_per_m2', 'reanalysis_relative_humidity_percent', 'reanalysis_sat_precip_amt_mm',
    'reanalysis_specific_humidity_g_per_kg', 'reanalysis_tdtr_k', 'station_avg_temp_c',
    'station_diur_temp_rng_c', 'station_max_temp_c', 'station_min_temp_c', 'station_precip_mm']

    path_train_results = "/Users/bhavish96.n/Documents/UTD/Fall '18/Machine Learning [Anurag Nagar]/Assignments/Long Project 1/Project Data/dengue_labels_train.csv"
    train_result_names = "city	year	weekofyear	total_cases".split()

    path_test = ""
    path_test_results = ""


    deng_object = dengAi(path_train, train_names, path_train_results, train_result_names, path_test, path_test_results)
    deng_object.preprocess_data()
    deng_object.decision_tree()
    #deng_object.svm()

    #deng_object.random_forest()

if __name__ == '__main__':
    main()
