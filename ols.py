import linear

def pre_processor(filename) :

    import pandas as pd

    dataset = pd.read_csv(filename)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    X = linear.cleaner(X)
    X = linear.dim_encoder(X,0)
    y = linear.y_encoder(y)

    X_train, X_test, y_train, y_test = linear.splitter(X,y)

    return X_train, X_test, y_train, y_test, X, y

#procs

X_train, X_test, y_train, y_test, X, y = pre_processor('Data.csv')
