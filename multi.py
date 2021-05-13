

def preprocess(filename) :

    import pandas as pd
    from sklearn.model_selection import train_test_split
    #import linear

    dataset = pd.read_csv(filename)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    #X = linear.dim_encoder(X,3)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

    return X_train, X_test, y_train, y_test


def regress(X_train,y_train) :

    from sklearn.linear_model import LinearRegression

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)

    return regressor


import numpy as np
from sklearn.metrics import r2_score
import sv_plus_trees as treeplus
import poly as poly


X_train, X_test, y_train, y_test = preprocess('Data.csv')
#X_train = treeplus.scale_feature(X_train)
#y_train = treeplus.scale_feature(y_train)

regressor = regress(X_train,y_train)
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)

score = r2_score(y_test, y_pred)
print("Multiple Linear")
print(score)

"""
print( np.concatenate(
    (
      y_pred.reshape(len(y_pred),1),
      y_test.reshape(len(y_test),1)
    ),
    1
  )
)
"""
