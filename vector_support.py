def preprocess(filename) :

    import pandas as pd

    dataset = pd.read_csv(filename)
    X = dataset.iloc[:, 1:-1].values
    y = dataset.iloc[:, -1].values

    return X, y

def scale_feature(X) :
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X = sc.fit_transform(X)

    return X, sc

def sv_regress(arg,X,y) :
    from sklearn.svm import SVR
    regressor = SVR(kernel = arg)
    regressor.fit(X,y)
    return regressor

def tree_regress(X,y) :
    from sklearn.tree import DecisionTreeRegressor
    regressor = DecisionTreeRegressor(random_state = 0)
    regressor.fit(X,y)
    return regressor

def scatter (X,y,line) :
    import matplotlib.pyplot as plt
    plt.scatter(X, y, color = 'red')
    plt.plot(X,line, color = 'blue')
    plt.title('Support Vector Regression')
    plt.xlabel('Title')
    plt.ylabel('Salary')
    plt.show()
    return True

def hidef_scatter (X,y,regressor) :
    import matplotlib.pyplot as plt
    import numpy as np
    X_grid = np.arange(min(X),max(X),0.1)
    X_grid = X_grid.reshape(len(X_grid),1)
    plt.scatter(X, y, color = 'red')
    plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
    plt.title('Decision Tree Regression')
    plt.xlabel('Title')
    plt.ylabel('Salary')
    plt.show()
    return True


X, y = preprocess('Position_Salaries.csv')

#Decision Tree Regression
regressor = tree_regress(X,y)

hidef_scatter(X,y,regressor)


#Support Vector Regression
"""
y = y.reshape(len(y),1)
X, sc_X = scale_feature(X)
y, sc_y = scale_feature(y)
"""

"""
regressor = sv_regress('rbf',X,y)

result = sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])))

scatter(
    sc_X.inverse_transform(X),
    sc_y.inverse_transform(y),
    sc_y.inverse_transform(regressor.predict(X))
)
"""
