def preprocess(filename) :

    import pandas as pd
    from sklearn.model_selection import train_test_split

    dataset = pd.read_csv(filename)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

    return X_train, X_test, y_train, y_test

def scale_feature(X) :
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test

X_train, X_test, y_train, y_test = preprocess('Social_Network_Ads.csv')
