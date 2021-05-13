def preprocess(filename) :

    import pandas as pd
    from sklearn.model_selection import train_test_split

    dataset = pd.read_csv(filename)
    X = dataset.iloc[:, 1:-1].values
    y = dataset.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    return X_train, X_test, y_train, y_test

def regress(X_train,y_train) :

    from sklearn.linear_model import LinearRegression

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)

    return regressor

def poly_regress(X,y) :

    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression

    poly_reg = PolynomialFeatures(degree = 2)
    X_poly = poly_reg.fit_transform(X)
    lin_reg = LinearRegression()
    lin_reg.fit(X_poly,y)

    return lin_reg, poly_reg


import numpy as np
from sklearn.metrics import r2_score
import linear

X_train, X_test, y_train, y_test = preprocess('Data.csv')

#regressor = regress(X_train,y_train)
p_lin_regressor, poly_regressor = poly_regress(X_train,y_train)
y_pred = p_lin_regressor.predict(poly_regressor.transform(X_test))


score = r2_score(y_test, y_pred)
print("Polynomial")
print(score)

#linear.scatter(X_train,y_train,p_lin_regressor,poly_regressor.fit_transform(X_train))
