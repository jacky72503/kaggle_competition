import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder

# get the relative path of dataset
absolutepath = os.path.abspath(__file__)
fileDirectory = os.path.dirname(absolutepath)
# load data
train = pd.read_csv(os.path.join(fileDirectory, 'train.csv'), index_col='id')
test = pd.read_csv(os.path.join(fileDirectory, 'test.csv'), index_col='id')

# print(train.shape)
# print(train.describe())
# print(train.head())
print(train['claim'].describe())
print(train['claim'].value_counts())
train.dropna(axis=0, subset=['claim'], inplace=True)
y_train = train.claim
train.drop(['claim'], axis=1, inplace=True)
print(train.shape)
X_train = train.copy()
X_test = test.copy()
# i = 0
# drop_rate = 0.7
# while i < train.shape[1]:
#     if (train.iloc[:, i].isnull().values.any()):
#         print("{0} idx is {1} that has {2} null values".format(train.columns[i], i, train.iloc[:, i].isnull().sum()))
#         if train.iloc[:, i].isnull().sum() / train.shape[0] > drop_rate:
#             print("drop {}".format(train.columns[i]))
#             train.drop(train.columns[i], axis=1, inplace=True)
#     i += 1
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, RandomizedSearchCV, \
    cross_val_score, KFold
from xgboost import XGBClassifier

# xgb_params = {
#     'n_estimators': 5000,
#     'learning_rate': 0.1235,
#     'subsample': 0.95,
#     'colsample_bytree': 0.11,
#     'max_depth': 2,
#     'booster': 'gbtree',
#     'reg_lambda': 66.1,
#     'reg_alpha': 15.9,
#     'random_state':42
# }
kf = KFold(n_splits = 3, shuffle = True, random_state = 42)
model3 = XGBClassifier()
results = cross_val_score(model3 ,X_train,y_train,cv=kf,scoring='roc_auc')
print(results)
print("score: {} {}%".format(results.mean,results.std))


