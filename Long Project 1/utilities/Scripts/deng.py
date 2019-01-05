# Utilities import
import numpy as np
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Models import
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Metrics import
from sklearn.metrics import mean_absolute_error, classification_report, accuracy_score

class dengAi():
	def __init__(self, train_path, train_names, train_results_path, train_result_names, path_test):
		self.X_train = pd.read_csv(
		train_path,
		)
		self.y_train = pd.read_csv(
		train_results_path,
		)
		self.correlation = 0
		self.final_columns = []

		self.x_vals = []
		self.y_vals = []

		self.init_models()

	def init_models(self):
		self.d_tree = DecisionTreeRegressor(
			criterion="mae", splitter="best", random_state=40
		)

		# Report
		self.decision_tree_report = []

		# Prediction values
		self.decision_tree_predictions = []




	def preprocess_data(self):
		X_train = pd.concat([self.X_train, self.y_train['total_cases']], axis=1)

		X_train.fillna(method='ffill', inplace=True)

		random_DF = X_train.reindex(np.random.permutation(X_train.index))

		encoder = LabelEncoder()
		random_DF["city"] = encoder.fit_transform(random_DF["city"])

		seasons = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0]
		seasons_series = []

		for item in random_DF['week_start_date']:
			seasons_series.append(
				seasons[
					datetime.strptime(item, "%Y-%m-%d").month - 1
					]
			)

		random_DF["seasons"] = pd.Series(seasons_series)
		random_DF = random_DF.drop("week_start_date", axis=1)

		random_DF.city = random_DF.city.astype("float64")
		random_DF.year = random_DF.year.astype("float64")
		random_DF.week_of_year = random_DF.weekofyear.astype("float64")
		random_DF.seasons = random_DF.seasons.astype("float64")

		print(random_DF)

		modified_y_train = random_DF['total_cases']
		modified_X_train = random_DF.drop('total_cases', axis=1)

		self.x_vals = modified_X_train.values
		self.y_vals = modified_y_train.values
		self.y_valsy_vals = self.y_vals.reshape(self.y_vals.shape[0], 1)

	# Decision tree
	def decision_tree_init(self):
		self.d_tree.fit(self.x_vals, self.y_vals)

	def decision_tree_predict(self):
		self.decision_tree_predictions = self.d_tree.predict(self.x_vals)
		self.decision_tree_predictions = self.decision_tree_predictions.astype(int)
		self.decision_tree_report = classification_report(
			self.y_vals, self.decision_tree_predictions,
			output_dict = True
		)

		print (self.decision_tree_report['weighted avg'])


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


	deng_object = dengAi(path_train, train_names, path_train_results, train_result_names, path_test)
	deng_object.preprocess_data()
	deng_object.decision_tree_init()
	deng_object.decision_tree_predict()

if __name__ == '__main__':
	main()


'''
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

class dengAi():
	def __init__(self, train_path, train_names, train_results_path, train_result_names, path_test, path_test_results):
		self.X_train = pd.read_csv(
		train_path, names = train_names, header = None, skiprows = 1, #index_col= [0, 1, 2]
		)
		self.y_train = pd.read_csv(
		train_results_path, names = train_result_names, header = None, skiprows = 1, #index_col= [0, 1, 2]
		)
		self.correlation = 0
		self.train_features = train_names
		self.columns = set(self.train_features) - set(['year', 'week_start_date', 'city', 'weekofyear'])
		self.final_columns = []
		self.iq_final_columns = []


	def find_columns(self, corr):
		final_columns = []
		corr_values = (
			corr.total_cases
				.drop('total_cases')  # don't compare with myself
				.sort_values(ascending=False)
		)

		# To get a variation of columns based on the correlation values
		corr_max = corr_values.max()
		corr_min = corr_values.min()


		corr_values = corr_values.between(0.10, 0.30, inclusive=True)

		for item in corr_values.index:
			if corr_values[item] == True:
				final_columns.append(item)
		return final_columns


	def preprocess_data(self):
		# Converting categorical data to numerical
		encoder = LabelEncoder()
		self.X_train["city"] = encoder.fit_transform(self.X_train["city"])
		print (self.X_train.head)

		# Removing col 3 from the training data
		self.X_train.drop('week_start_date', axis=1, inplace=True)

		# Check for Null values
		#if pd.isnull(self.X_train).any().all():
		self.X_train.fillna(method='ffill', inplace=True)

		# Appending total_cases to the sj_train_features and iq_train_features dataframe to compute correlation
		self.X_train['total_cases'] = self.y_train.total_cases

		self.correlation = self.X_train.corr()

		# Plot the maps with a new function

		# Get the relevant columns
		self.final_columns = self.find_columns(self.correlation)

		# print (self.sj_final_columns, self.iq_final_columns)

		''''''
		self.sj_train_features = self.X_train.loc['sj']
		self.sj_train_labels = self.y_train.loc['sj']

		self.iq_train_features = self.X_train.loc['iq']
		self.iq_train_labels = self.y_train.loc['iq']

		# Removing col 3 from the training data
		self.sj_train_features.drop('week_start_date', axis=1, inplace=True)
		self.iq_train_features.drop('week_start_date', axis=1, inplace=True)

		# Check for Null values
		if pd.isnull(self.sj_train_features).any().all():
			self.sj_train_features.fillna(method = 'ffill', inplace = True)
		if pd.isnull(self.iq_train_features).any().all():
			self.iq_train_features.fillna(method = 'ffill', inplace = True)

		# Appending total_cases to the sj_train_features and iq_train_features dataframe to compute correlation
		self.sj_train_features['total_cases'] = self.sj_train_labels.total_cases
		self.iq_train_features['total_cases'] = self.iq_train_labels.total_cases

		# Correlation
		self.sj_corr = self.sj_train_features.corr()
		self.iq_corr = self.iq_train_features.corr()

		# Plot the maps with a new function

		# Get the relevant columns
		self.sj_final_columns = self.find_columns(self.sj_corr)
		self.iq_final_columns = self.find_columns(self.iq_corr)

		#print (self.sj_final_columns, self.iq_final_columns)
		''''''

	def decision_trees(self):
		train_data = pd.DataFrame()
		for column in self.final_columns:
			train_data[column] = self.X_train[column]
		train_lables = self.y_train['total_cases']

		X_train, X_test, y_train, y_test = train_test_split(
			train_data, train_lables, test_size = 0.33, random_state= 42
		)

		dengTreeModel = tree.DecisionTreeRegressor(criterion="mae", splitter="best", random_state=40)
		dengTreeModel.fit(X_train, y_train)

		test_preds_tree = dengTreeModel.predict(X_test)
		print(mean_absolute_error(y_test, test_preds_tree))
		print (accuracy_score(y_test, test_preds_tree))

		print (
			max(recall_score(y_test, test_preds_tree, average = None)),
			max(f1_score(y_test, test_preds_tree, average = None)),
			max(precision_score(y_test, test_preds_tree, average = None))
		)


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


	deng_object = dengAi(path_train, train_names, path_train_results, train_result_names, path_test, path_test_results)
	deng_object.preprocess_data()
	deng_object.decision_trees()


if __name__ == '__main__':
	main()
'''
