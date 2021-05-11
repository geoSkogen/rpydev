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

def random_forest_regress(X,y) :
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
    regressor.fit(X,y)
    return regressor

def scatter (X,y,line,title) :
    import matplotlib.pyplot as plt
    plt.scatter(X, y, color = 'red')
    plt.plot(X,line, color = 'blue')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.show()
    return True

def hidef_scatter (X,y,regressor,title) :
    import matplotlib.pyplot as plt
    import numpy as np
    X_grid = np.arange(min(X),max(X),0.1)
    X_grid = X_grid.reshape(len(X_grid),1)
    plt.scatter(X, y, color = 'red')
    plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.show()
    return True

def support_vector_regress(X,y) :
    y = y.reshape(len(y),1)
    X, sc_X = scale_feature(X)
    y, sc_y = scale_feature(y)

    regressor = sv_regress('rbf',X,y)

    result = sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])))

    scatter(
        sc_X.inverse_transform(X),
        sc_y.inverse_transform(y),
        sc_y.inverse_transform(regressor.predict(X)),
        'Support Vector Regression'
    )
    return True


X, y = preprocess('Position_Salaries.csv')

"""

regressor = tree_regress(X,y)
title = 'Decision Tree Regression'

"""

regressor = random_forest_regress(X,y)
title = 'Random Forest Regression'


hidef_scatter(X,y,regressor,title)


#Support Vector Regression
