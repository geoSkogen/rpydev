import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

def ols_regress(X,y) :

    regressor = LinearRegression()
    regressor.fit(X, y)
    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)

    return regressor

def scatter (X,y,lin_reg,poly_X) :
    plt.scatter(X, y, color = 'red')
    plt.plot(X,lin_reg.predict(poly_X), color = 'blue')
    plt.title('Regression Line / Scaatter Plot')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.show()
    return True

def cleaner (input_table) :
    imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
    imputer.fit(input_table[:,1:3])
    input_table[:,1:3] = imputer.transform(input_table[:,1:3])
    return input_table

def dim_encoder (input_table,col_index) :
    ct = ColumnTransformer(
        transformers = [ ('encoder', OneHotEncoder(), [col_index]) ],
        remainder='passthrough'
        )
    input_table = np.array(ct.fit_transform(input_table))
    return input_table

def y_encoder (input_arr) :
    le = LabelEncoder()
    output_arr = le.fit_transform(input_arr)
    return output_arr

def splitter (X_matrix,y_arr) :

    X_train, X_test, y_train, y_test = train_test_split(X_matrix, y_arr, test_size = 0.2, random_state = 1)

    sc = StandardScaler()
    X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
    X_test[:, 3:] = sc.transform(X_test[:, 3:])

    return X_train, X_test, y_train, y_test
