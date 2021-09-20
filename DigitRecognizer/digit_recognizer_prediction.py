import os
import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

def plt_data_bar(series):
    X = series.value_counts().index
    Y = series.value_counts().values
    print(X,Y)
    fig, ax = plt.subplots(figsize=(10, 4))
    idx = np.asarray([i for i in range(len(X))])
    data = {key: val for key, val in zip(X, Y)}
    ax.bar(idx, [val for key,val in sorted(data.items())])
    ax.set_xticks(idx)
    ax.set_xticklabels(X)
    ax.set_xlabel(series.name)
    ax.set_ylabel('count')
    plt.show()

absolutepath = os.path.abspath(__file__)
fileDirectory = os.path.dirname(absolutepath)

train = pd.read_csv(fileDirectory + r'\train.csv')
test = pd.read_csv(fileDirectory + r'\test.csv')
submission = pd.read_csv(fileDirectory + r'\sample_submission.csv')

X_train, X_val, y_train, y_val = train_test_split(train.drop('label',axis=1), train['label'], test_size=0.1, random_state=33)

xgb_model = XGBClassifier(objective='multi:softprob',
                      num_class= 10)
xgb_model.fit(X_train, y_train)

preds = xgb_model.predict(test).astype(int)
save = pd.DataFrame({'ImageId':submission.ImageId,'label':preds})
save.to_csv('submission.csv',index=False)