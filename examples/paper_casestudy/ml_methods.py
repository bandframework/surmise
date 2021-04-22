from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import confusion_matrix
import numpy as np

def fit_RandomForest(func_eval, param_values, T0, T1):
    # Run a classification model
    y = np.zeros(len(func_eval))
    y[np.logical_and.reduce((func_eval[:, 25] < 350, func_eval[:, 100] > T0, func_eval[:, 100] < T1))] = 1
    
    print(np.sum(y))
    # Create a balanced data set
    split = 200
    X_0 = param_values[y == 0][0:split]
    y_0 = y[y == 0][0:split]
    X_1 = param_values[y == 1][0:split]
    y_1 = y[y == 1][0:split]
    
    X_0test = param_values[y == 0][split:(2*split)]
    y_0test = y[y == 0][split:(2*split)]
    X_1test = param_values[y == 1][split:(2*split)]
    y_1test = y[y == 1][split:(2*split)]
    
    X = np.concatenate((X_0, X_1), axis=0)
    y = np.concatenate((y_0, y_1), axis=0)
    
    Xtest = np.concatenate((X_0test, X_1test), axis=0)
    ytest = np.concatenate((y_0test, y_1test), axis=0)
    
    # Fit the classification model
    model = RandomForestClassifier(n_estimators = 100, random_state = 42)
    model.fit(X, y)
    
    # Training accuracy
    print(model.score(X, y))
    print(confusion_matrix(y, model.predict(X)))
    
    # Test accuracy
    print(confusion_matrix(ytest, model.predict(Xtest)))
    print(model.score(Xtest, ytest))
    
    return model

def fit_logisticRegression(func_eval, param_values, T0, T1):
    # Run a classification model
    y = np.zeros(len(func_eval))
    y[np.logical_and.reduce((func_eval[:, 25] < 350, func_eval[:, 100] > T0, func_eval[:, 100] < T1))] = 1
    
    print(np.sum(y))
    # Create a balanced data set
    split = 300
    X_0 = param_values[y == 0][0:split]
    y_0 = y[y == 0][0:split]
    X_1 = param_values[y == 1][0:split]
    y_1 = y[y == 1][0:split]
    
    X_0test = param_values[y == 0][split:(split+150)]
    y_0test = y[y == 0][split:(split+150)]
    X_1test = param_values[y == 1][split:(split+150)]
    y_1test = y[y == 1][split:(split+150)]
    
    X = np.concatenate((X_0, X_1), axis=0)
    y = np.concatenate((y_0, y_1), axis=0)
    
    Xtest = np.concatenate((X_0test, X_1test), axis=0)
    ytest = np.concatenate((y_0test, y_1test), axis=0)
    
    for i in range(0, 4):
        for j in range(i, 4):
            X = np.concatenate([X, np.reshape(X[:, i] * X[:, j], (len(X), 1))], axis = 1)
            
    for i in range(0, 4):
        for j in range(i, 4):
            Xtest = np.concatenate([Xtest, np.reshape(Xtest[:, i] * Xtest[:, j], (len(Xtest), 1))], axis = 1)
    
    # Fit the classification model
    #model = RandomForestClassifier(n_estimators = 100, random_state = 42)
    model = LogisticRegression(fit_intercept=True, penalty = 'none', max_iter = 50000, solver = 'saga', tol = 1e-4)
    model.fit(X, y)
    
    # Training accuracy
    print(model.score(X, y))
    print(confusion_matrix(y, model.predict(X)))
    
    # Test accuracy
    print(confusion_matrix(ytest, model.predict(Xtest)))
    print(model.score(Xtest, ytest))
    
    return model