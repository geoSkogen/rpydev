def preprocess(filename) :

    import pandas as pd
    from sklearn.model_selection import train_test_split

    dataset = pd.read_csv(filename)
    X = dataset.iloc[:, 1:-1].values
    y = dataset.iloc[:, -1].values

    y = y.reshape(len(y),1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    return X_train, X_test, y_train, y_test

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

def support_vector_regress(X,y,X_test) :
    y = y.reshape(len(y),1)
    X, sc_X = scale_feature(X)
    y, sc_y = scale_feature(y)

    regressor = sv_regress('rbf',X,y)

    y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)))
    """
    scatter(
        sc_X.inverse_transform(X),
        sc_y.inverse_transform(y),
        sc_y.inverse_transform(regressor.predict(X)),
        'Support Vector Regression'
    )
    """
    return regressor, y_pred

from sklearn.metrics import r2_score

X_train, X_test, y_train, y_test = preprocess('Data.csv')

tree_regressor = tree_regress(X_train,y_train)
tree_y_pred = tree_regressor.predict(X_test)
#title = 'Decision Tree Regression'

forest_regressor = random_forest_regress(X_train,y_train)
forest_y_pred = forest_regressor.predict(X_test)
#title = 'Random Forest Regression'

#hidef_scatter(X_train,y_train,regressor,title)

sv_regressor, sv_y_pred = support_vector_regress(X_train,y_train,X_test)
print("Supoort Vector")
r2_score_sv = r2_score(y_test, sv_y_pred)
print(r2_score_sv)
print('Decision Tree')
r2_score_tree = r2_score(y_test, tree_y_pred)
print(r2_score_tree)
print('Random Forest')
r2_score_forest = r2_score(y_test, forest_y_pred)
print(r2_score_forest)
