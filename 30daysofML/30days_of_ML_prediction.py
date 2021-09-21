import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, RandomizedSearchCV, \
    cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

absolutepath = os.path.abspath(__file__)
fileDirectory = os.path.dirname(absolutepath)
# read data
train = pd.read_csv(fileDirectory + r'\train.csv', index_col='id')
test = pd.read_csv(fileDirectory + r'\test.csv', index_col='id')
submission = pd.read_csv(fileDirectory + r'\sample_submission.csv')
# print(test.head())
# print(train.shape)
# print(train.head())
# print(train.columns)

# Remove rows with missing target, separate target from predictors
train.dropna(axis=0, subset=['target'], inplace=True)
y_train = train.target
train.drop(['target'], axis=1, inplace=True)
print(train.shape)
X_train = train.copy()
X_test = test.copy()
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2,
#                                                     random_state=0)

# # Select categorical columns with relatively low cardinality (convenient but arbitrary)
# low_cardinality_cols = [cname for cname in X_train.columns if X_train[cname].nunique() < 20 and
#                         X_train[cname].dtype == "object"]
# print(low_cardinality_cols)
# # Select numeric columns
# numeric_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]
# print(numeric_cols)
#
# # Keep selected columns only
# my_cols = low_cardinality_cols + numeric_cols
# X_train = X_train[my_cols].copy()
# X_valid = X_valid[my_cols].copy()
# X_test = test[my_cols].copy()

# One-hot encode the data (to shorten the code, we use pandas)

# model1
# model1 = XGBRegressor()
# model1.fit(X_train, y_train)
# pred1 = model1.predict(X_test)
# mae1 = mean_absolute_error(pred1, y_test)
# mse1 = mean_squared_error(pred1, y_test)
# print("Mean Absolute Error:" , mae1)
# print("Mean Squared Error:" , mse1)

# model2
# cv_params = {'n_estimators': np.arange(200, 600, 50),
#              'eta': np.arange(0.1, 0.3, 0.04),
#              'min_child_weight': np.arange(1, 3, 1),}
# other_params = {'max_depth':3,'n_estimators':550,'min_child_weight':1,'eta':0.2},
# model2 = xgb.XGBRegressor(**other_params)
# random_search = RandomizedSearchCV(estimator=model2, param_distributions=cv_params, scoring='neg_mean_squared_error', cv=5, n_iter=10,
#                     verbose=3, n_jobs=4,random_state=1001)
# random_search.fit(X_train, y_train)
# print('best_params：{0}'.format(random_search.best_params_))
# print('best_score:{0}'.format(random_search.best_score_))

# model3
from sklearn.preprocessing import  OrdinalEncoder
category_cols = [col for col in train.columns if 'cat' in col]
enc = OrdinalEncoder()
X_train[category_cols] = enc.fit_transform(train[category_cols])
X_test[category_cols] = enc.transform(X_test[category_cols])
X = train.copy

xgb_params = {
    'n_estimators': 5000,
    'learning_rate': 0.1235,
    'subsample': 0.95,
    'colsample_bytree': 0.11,
    'max_depth': 2,
    'booster': 'gbtree',
    'reg_lambda': 66.1,
    'reg_alpha': 15.9,
    'random_state':42
}
kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
model3 = XGBRegressor(**xgb_params)
results = cross_val_score(model3 ,X_train,y_train,cv=kf,scoring='neg_root_mean_squared_error')
print(results)
print("score: {} {}%".format(results.mean,results.std))





