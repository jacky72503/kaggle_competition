import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import cross_val_score, KFold
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)


def preprocessing(df):
    df[['group', 'id']] = df['PassengerId'].str.split('_', 1, expand=True)
    df[['deck', 'num', 'side']] = df['Cabin'].str.split('/', expand=True)
    df = df.drop(columns=['PassengerId', 'Cabin', 'Name', 'num', 'group', 'id'])
    # drop all null rows
    # df = df.dropna()

    cat_cols = ['HomePlanet', 'Destination', 'deck', 'side']
    bool_col = ['CryoSleep', 'VIP']

    le = LabelEncoder()
    df[bool_col + cat_cols] = df[bool_col + cat_cols].apply(le.fit_transform)

    ohe = OneHotEncoder()
    df_cat_ohe = pd.DataFrame(ohe.fit_transform(df[cat_cols]).toarray(), index=df.index)
    df = df.drop(columns=cat_cols)
    df = pd.concat([df, df_cat_ohe], axis=1)
    return df


data_root = "/media/jacky72503/data/spaceship-titanic"
train_df = pd.read_csv(f"{data_root}/train.csv")
train_df = preprocessing(train_df)

y_train = LabelEncoder().fit_transform(train_df['Transported'])
X_train = train_df.drop(columns=['Transported'])

kf = KFold(n_splits=10, shuffle=True, random_state=42)
model = XGBClassifier(eval_metric=['logloss', 'auc', 'error'])
results = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')
model.fit(X_train, y_train)
print(model.feature_importances_)
print(results)
print("mean score: {}".format(np.mean(results)))

test_df = pd.read_csv(f"{data_root}/test.csv")
submission_id=test_df['PassengerId']
test_df = preprocessing(test_df)
pred = model.predict(test_df)
mapping = np.vectorize(lambda x: "True" if x == 1 else "False")

submission = pd.DataFrame({'PassengerId': submission_id, 'Transported': mapping(pred)})
submission.to_csv("submission_spaceship_titanic.csv",index=False)

# df1 = pd.DataFrame()
# df1[['group','id']] = df['PassengerId'].str.split('_', 1, expand=True)
# df1['Transported'] = df['Transported']
# print(len(df1))
# print(len(df1['group'].unique()))
# print(len(df1.drop_duplicates(subset=['group','Transported'])))
# print((len(df1)-len(df1.drop_duplicates(subset=['group','Transported'])))/(len(df1)-len(df1['group'].unique())))
