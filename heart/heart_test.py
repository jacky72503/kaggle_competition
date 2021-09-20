import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from xgboost import XGBClassifier
from matplotlib import pyplot as plt
pd.set_option('display.max_columns', None)


def OH_encoder(df, column_name):
    ohenc = OneHotEncoder(sparse=False)
    result = ohenc.fit_transform(pd.DataFrame(df[column_name]))
    fn = ohenc.get_feature_names([column_name])
    ohe_df = pd.DataFrame(result, columns=fn)
    ohe_df.set_index(np.arange(0, ohe_df.shape[0], 1), inplace=True)
    df = pd.concat([df, ohe_df], axis=1).drop([column_name], axis=1)
    return df


def ordinal_encoder(df, column_name, cat):
    ordenc = OrdinalEncoder(categories=[cat])
    result = ordenc.fit_transform(pd.DataFrame(df[column_name]))
    df[column_name] = pd.DataFrame(result)
    return df


# get the relative path of dataset
absolutepath = os.path.abspath(__file__)
fileDirectory = os.path.dirname(absolutepath)
# load data
data = pd.read_csv(os.path.join(fileDirectory, 'heart.csv'))
print(data.shape)
print(data.columns)
data = OH_encoder(data, 'Sex')
data = ordinal_encoder(data, 'ChestPainType', ['TA', 'ATA', 'NAP', 'ASY'])
data = OH_encoder(data, 'RestingECG')
data = OH_encoder(data, 'ExerciseAngina')
data = OH_encoder(data, 'ST_Slope')
y_train = data.HeartDisease
X_train = data.drop('HeartDisease', axis=1)
# print(X_train)
# print(y_train)

from sklearn.model_selection import GridSearchCV,cross_val_score, KFold
kf = KFold(n_splits = 10, shuffle = True, random_state = 42)
model = XGBClassifier()
results = cross_val_score(model ,X_train,y_train,cv=kf,scoring='accuracy')
model.fit(X_train,y_train)
print(results)
print("mean score: {}".format(np.mean(results)))
print(model.feature_importances_)
plt.barh(X_train.columns,model.feature_importances_)
plt.show()
# numerical_columns = data.select_dtypes(include=['float64','int64']).columns
# numerical_data = data[numerical_columns]
# print(numerical_data)
