import numpy as np
import pandas as pd


class LinearRegression:
    def __init__(self, X:pd.DataFrame, y:pd.DataFrame, method='gradient descent'):
        '''
        '''
        assert isinstance(X, pd.DataFrame) # We expect the data to be pd.Dataframe
        assert isinstance(y, pd.DataFrame) # We expect the data to be pd.Dataframe

        features = X.columns
        self.target_name = y.columns[0]
        X = np.array(X.values).astype(float)

        # We add a column of ones to the features matrix
        self.X = np.column_stack((np.ones(X.shape[0]), X))
        self.y = np.array(y.values).flatten()

        self.features = np.array(['target'] + list(features))
        self.coef = np.zeros(len(self.features))
        self.coef_trace = [self.coef.copy()]
        
        if method == 'gradient descent':
            self.gradient_descent(self.coef.copy())
        elif method == 'pseudoinverse':
            self.pseudoinverse()
        else:
            raise Exception('Invalid training method.')
        
    def loss(self, w):
        n = len(self.y)
        residuals = self.X @ w - self.y
        return (1/n) * (residuals.T @ residuals)

    def gradient(self, w):
        n = len(self.y)
        return 2/n * self.X.T @ ((self.X @ w) - self.y)
    
    def gradient_descent(self, x0, lr=0.01, tol=1e-10, max_iter=50000):
        xk = x0
        self.coef_trace = [x0]
        k = 0

        while np.linalg.norm(self.gradient(xk)) > tol and k < max_iter:
            xk = xk - lr*self.gradient(xk)
            self.coef_trace.append(xk.copy())
            k += 1

        self.coef = xk
        return xk
    
    def pseudoinverse(self):
        params = np.linalg.pinv(self.X.T @ self.X) @ self.X.T @ self.y
        self.coef = params
        return params
    
    def predict(self, X_new):
        X_new = np.asarray(X_new)
        X_new = np.column_stack((np.ones(X_new.shape[0]), X_new))
        return X_new @ self.coef
    
    def __str__(self):
        eq = f'{self.target_name} = '
        bias = f'{self.coef[0]:.2f}'
        eq += bias
        for feature, weight in zip(self.features[1:], self.coef[1:]):
            eq += f' + ({weight:.2f} * {feature})'
        return eq
    