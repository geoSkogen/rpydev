def preprocess(filename) :

    import pandas as pd
    from sklearn.model_selection import train_test_split
    import linear

    dataset = pd.read_csv(filename)
    X = dataset.iloc[:, 1:-1].values
    y = dataset.iloc[:, -1].values

    return X, y

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
import linear

X,y = preprocess('Position_Salaries.csv')
regressor = regress(X,y)
p_lin_regressor, poly_regressor = poly_regress(X,y)

linear.scatter(X,y,p_lin_regressor,poly_regressor.fit_transform(X))
