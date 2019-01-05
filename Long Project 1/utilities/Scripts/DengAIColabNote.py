
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score
from datetime import datetime
from sklearn import tree


from sklearn.ensemble import RandomForestRegressor
# load the provided data
dengue_features_df = pd.read_csv("/Users/bhavish96.n/Documents/UTD/Fall '18/Machine Learning [Anurag Nagar]/Assignments/Long Project 1/Project Data/dengue_features_train.csv")
dengue_feature_labels = pd.read_csv("/Users/bhavish96.n/Documents/UTD/Fall '18/Machine Learning [Anurag Nagar]/Assignments/Long Project 1/Project Data/dengue_labels_train.csv")
dengue_features_df = pd.concat([dengue_features_df, dengue_feature_labels["total_cases"]], axis=1)

'''
dengue_features_df["ndvi_ne"] = dengue_features_df["ndvi_ne"].fillna(dengue_features_df["ndvi_ne"].mean())
dengue_features_df["ndvi_nw"] = dengue_features_df["ndvi_nw"].fillna(dengue_features_df["ndvi_nw"].mean())
dengue_features_df["ndvi_se"] = dengue_features_df["ndvi_se"].fillna(dengue_features_df["ndvi_se"].mean())
dengue_features_df["ndvi_sw"] = dengue_features_df["ndvi_sw"].fillna(dengue_features_df["ndvi_sw"].mean())
'''

dengue_features_df.fillna(method='ffill', inplace=True)

print(dengue_features_df.isnull().sum())

new_den_fea_df = dengue_features_df.reindex(np.random.permutation(dengue_features_df.index))

encoder = LabelEncoder()
new_den_fea_df["city"] = encoder.fit_transform(new_den_fea_df["city"])


seasons = [0,0,1,1,1,2,2,2,3,3,3,0]
new_den_fea_df["seasons"] = new_den_fea_df["week_start_date"].apply(lambda x: seasons[(datetime.strptime(x, "%Y-%m-%d").month-1)])

new_den_fea_df = new_den_fea_df.drop("week_start_date", axis=1)



new_den_fea_df.city = new_den_fea_df.city.astype("float64")
new_den_fea_df.year = new_den_fea_df.year.astype("float64")
new_den_fea_df.weekofyear = new_den_fea_df.weekofyear.astype("float64")
new_den_fea_df.seasons = new_den_fea_df.seasons.astype("float64")



y_df_vals = new_den_fea_df["total_cases"].copy()
x_df_vals = new_den_fea_df.drop("total_cases", axis=1)

x_vals = x_df_vals.values
y_vals = y_df_vals.values

y_vals = y_vals.reshape(y_vals.shape[0], 1)

dengTreeModel = tree.DecisionTreeRegressor(criterion="mae", splitter="best", random_state=40)
dengTreeModel.fit(x_vals, y_vals)

den_test_df = pd.read_csv("/Users/bhavish96.n/Documents/UTD/Fall '18/Machine Learning [Anurag Nagar]/Assignments/Long Project 1/Project Data/dengue_features_test.csv")
den_test_df.info()

den_test_df["seasons"] = den_test_df["week_start_date"].apply(lambda x : seasons[datetime.strptime(x, "%Y-%m-%d").month-1])

den_test_df["city"][den_test_df["city"] == "sj"] = 1
den_test_df["city"][den_test_df["city"] == "iq"] = 0

den_test_df.city = den_test_df.city.astype("float64")
den_test_df.year = den_test_df.year.astype("float64")
den_test_df.weekofyear = den_test_df.weekofyear.astype("float64")
den_test_df.seasons = den_test_df.seasons.astype("float64")

den_test_df = den_test_df.drop("week_start_date", axis=1)

print(den_test_df.isnull().sum())

den_test_df["ndvi_ne"] = den_test_df["ndvi_ne"].fillna(den_test_df["ndvi_ne"].mean())
den_test_df["ndvi_nw"] = den_test_df["ndvi_nw"].fillna(den_test_df["ndvi_nw"].mean())
den_test_df["ndvi_se"] = den_test_df["ndvi_se"].fillna(den_test_df["ndvi_se"].mean())
den_test_df["ndvi_sw"] = den_test_df["ndvi_sw"].fillna(den_test_df["ndvi_sw"].mean())

den_test_df.fillna(method='ffill', inplace=True)

den_test_preds = dengTreeModel.predict(x_df_vals.values)


den_test_preds = den_test_preds.astype(int)
print(mean_absolute_error(y_vals, den_test_preds))
print("Accuracy DecisionTreeRegressor %s" % accuracy_score(y_vals, dengTreeModel.predict(x_vals)))

den_test_preds = den_test_preds.reshape(den_test_preds.shape[0], 1)
print(den_test_preds.shape)

dengSVCModel = SVC(decision_function_shape="ovr", probability=True)
dengSVCModel.fit(x_vals, y_vals)

print(mean_absolute_error(y_vals, dengSVCModel.predict(x_vals)))

print("Accuracy SVC %s" % accuracy_score(y_vals, dengSVCModel.predict(x_vals)))

test_preds_svc = dengSVCModel.predict(den_test_df)
test_preds_tree = dengTreeModel.predict(den_test_df)

dengRFCModel = RandomForestRegressor(criterion="mae", random_state=40)
dengRFCModel.fit(x_vals, y_vals)
print("Accuracy RandomForestRegressor %s" % accuracy_score(y_vals, dengSVCModel.predict(x_vals)))

print(dengRFCModel.score(x_vals, y_vals))

print(dengTreeModel.score(x_vals, y_vals))

print(dengSVCModel.score(x_vals, y_vals))

test_preds_rfc = dengRFCModel.predict(den_test_df)

plt.scatter(dengSVCModel.predict(x_vals), y_vals, c="red")
plt.scatter(dengTreeModel.predict(x_vals), y_vals, c="blue")
plt.scatter(dengRFCModel.predict(x_vals), y_vals, c="green")
plt.title("Model Predictions")
plt.xlabel("predicted_cases")
plt.ylabel("actual cases")

ori_test_df = pd.read_csv("/Users/bhavish96.n/Documents/UTD/Fall '18/Machine Learning [Anurag Nagar]/Assignments/Long Project 1/Project Data/dengue_features_test.csv")
ori_test_df = ori_test_df.loc[:,["city", "year", "weekofyear"]]

test_preds_svc_df = pd.DataFrame(data = test_preds_svc, columns=["total_cases"])
test_preds_rfc_df = pd.DataFrame(data=test_preds_rfc, columns = ["total_cases"])
test_preds_tree_df = pd.DataFrame(data=test_preds_tree, columns=["total_cases"])

svc_gen_csv = pd.concat([ori_test_df, test_preds_svc_df], axis=1)
rfc_gen_csv = pd.concat([ori_test_df, test_preds_rfc_df], axis=1)
tree_gen_csv = pd.concat([ori_test_df, test_preds_tree_df], axis=1)

svc_gen_csv.total_cases = svc_gen_csv.total_cases.astype(int)
'''
print(svc_gen_csv.head())

rfc_gen_csv.total_cases = rfc_gen_csv.total_cases.astype(int)
tree_gen_csv.total_cases = tree_gen_csv.total_cases.astype(int)

rfc_gen_csv.head()

print(tree_gen_csv.head())

print(den_test_df.head())

svc_gen_csv.to_csv("C:\\Users\\Jay\\Documents\\UTD\\1st Sem\\ML\\Project\\svc_preds.csv", sep=",", index=False)
rfc_gen_csv.to_csv("C:\\Users\\Jay\\Documents\\UTD\\1st Sem\\ML\\Project\\rfc_preds.csv", sep=",", index=False)
tree_gen_csv.to_csv("C:\\Users\\Jay\\Documents\\UTD\\1st Sem\\ML\\Project\\tree_preds.csv", sep=",", index=False)
'''