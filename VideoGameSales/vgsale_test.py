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


# get the relative path of dataset
absolutepath = os.path.abspath(__file__)
fileDirectory = os.path.dirname(absolutepath)
# load data
data = pd.read_csv(os.path.join(fileDirectory, 'vgsales.csv'))
print(data.shape)
print(data.columns)

# # EDA
# sales = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
# # Distribution of platforms used by countries
# sales_by_platforms = data.groupby('Platform').sum()
# sales_by_platforms = sales_by_platforms[sales]
# sales_by_platforms = sales_by_platforms[sales_by_platforms.Global_Sales > 15]
# print(sales_by_platforms)
#
# fig, ax = plt.subplots(2,1,figsize=(15, 8))
#
# ax[0].bar(sales_by_platforms.index.values, sales_by_platforms['NA_Sales'].values, 0.35, label='NA_Sales')
# ax[0].bar(sales_by_platforms.index.values, sales_by_platforms['EU_Sales'].values, 0.35,
#        bottom=sales_by_platforms['NA_Sales'].values,
#        label='EU_Sales')
# ax[0].bar(sales_by_platforms.index.values, sales_by_platforms['JP_Sales'].values, 0.35,
#        bottom=sales_by_platforms['EU_Sales'].values + sales_by_platforms['NA_Sales'].values, label='JP_Sales')
# ax[0].bar(sales_by_platforms.index.values, sales_by_platforms['Other_Sales'].values, 0.35,
#        bottom=sales_by_platforms['JP_Sales'].values + sales_by_platforms['EU_Sales'].values + sales_by_platforms[
#            'NA_Sales'].values, label='Other_Sales')
# for i in range(len(sales_by_platforms['Global_Sales'])):
#     ax[0].text(i, int(sales_by_platforms['Global_Sales'].values[i])+5, int(sales_by_platforms['Global_Sales'].values[i]), ha='center')
#
# ax[0].set_ylabel('Sales')
# ax[0].set_title('sales by platforms')
# ax[0].legend()
# # Distribution of genre used by countries
# sales_by_genre = data.groupby('Genre').sum()
# print(sales_by_genre[sales])
# ax[1].bar(sales_by_genre.index.values, sales_by_genre['NA_Sales'].values, 0.35, label='NA_Sales')
# ax[1].bar(sales_by_genre.index.values, sales_by_genre['EU_Sales'].values, 0.35,
#        bottom=sales_by_genre['NA_Sales'].values,
#        label='EU_Sales')
# ax[1].bar(sales_by_genre.index.values, sales_by_genre['JP_Sales'].values, 0.35,
#        bottom=sales_by_genre['EU_Sales'].values + sales_by_genre['NA_Sales'].values, label='JP_Sales')
# ax[1].bar(sales_by_genre.index.values, sales_by_genre['Other_Sales'].values, 0.35,
#        bottom=sales_by_genre['JP_Sales'].values + sales_by_genre['EU_Sales'].values + sales_by_genre[
#            'NA_Sales'].values, label='Other_Sales')
# for i in range(len(sales_by_genre['Global_Sales'])):
#     ax[1].text(i, int(sales_by_genre['Global_Sales'].values[i])+5, int(sales_by_genre['Global_Sales'].values[i]), ha='center')
# ax[1].set_ylabel('Sales')
# ax[1].set_title('sales by genre')
# ax[1].legend()
# plt.show()


# preprocessing

from sklearn.impute import SimpleImputer
imp_most_frequent = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
data['Year'] = imp_most_frequent.fit_transform(data['Year'].values.reshape(-1,1))[:,0]
i = 0
drop_rate = 0.7
while i < data.shape[1]:
    if (data.iloc[:, i].isnull().values.any()):
        print("{0} idx is {1} that has {2} null values".format(data.columns[i], i, data.iloc[:, i].isnull().sum()))
        if data.iloc[:, i].isnull().sum() / data.shape[0] > drop_rate:
            print("drop {}".format(data.columns[i]))
            data.drop(data.columns[i], axis=1, inplace=True)
    i += 1
data.drop('Rank', axis=1, inplace=True)
data.drop('Name', axis=1, inplace=True)
data.drop('Publisher', axis=1, inplace=True)
data = OH_encoder(data, 'Platform')
data = OH_encoder(data, 'Genre')
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, KFold
train_columns = [col for col in data.columns if not ('Sales' in col)]
X = data[train_columns]
print(train_columns)
model_NA = XGBRegressor()
model_EU = XGBRegressor()
model_JP = XGBRegressor()
model_other = XGBRegressor()
model_Global = XGBRegressor()
def model_result(X,model,kfcv,target_column):
    results = cross_val_score(model, X, data[target_column].values, cv=kfcv, scoring='neg_mean_squared_error')
    print(results)
    print("{} score: {}".format(target_column,np.nanmean(results)))
kf = KFold(n_splits = 10, shuffle = True, random_state = 42)
model_result(X,model_NA,kf,'NA_Sales')
model_result(X,model_EU,kf,'EU_Sales')
model_result(X,model_JP,kf,'JP_Sales')
model_result(X,model_other,kf,'Other_Sales')
model_result(X,model_Global,kf,'Global_Sales')





