import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# SACAR COMENTARIOS CLASES Y EMPROLIJAR

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
    
class BinaryClassifier:
    def __init__(self, dataset, target_name='y', ridge_lambda=0, threshold=0.5, fit=True):
        assert isinstance(dataset, pd.DataFrame)  # Ensure the dataset is a pandas DataFrame

        # Separate features and target
        dataset_X = dataset.drop(columns=[target_name])
        X = np.array(dataset_X.values).astype(float)

        # Add a column of ones for the intercept
        self.X = np.column_stack((np.ones(X.shape[0]), X))
        self.y = np.array(dataset[target_name].values)

        self.features = np.array(['intercept'] + list(dataset_X.columns))
        self.target_name = target_name
        self.w = np.zeros(len(self.features))  # Initialize weights to zeros
        self.w_trace = [self.w.copy()]  # Track weight updates
        self.ridge_lambda = ridge_lambda
        self.threshold = threshold

        if fit:
            self.fit()

    def sigmoid(self, z):
        """Apply the sigmoid function."""
        return 1 / (1 + np.exp(-z))
    
    def loss(self, w):
        """Compute the log-loss with Ridge regularization."""
        n = len(self.y)
        predictions = self.sigmoid(self.X @ w)
        log_loss = -(1 / n) * (self.y @ np.log(predictions) + (1 - self.y) @ np.log(1 - predictions))

        # Add Ridge regularization term (excluding the intercept term)
        regularization_term = (self.ridge_lambda / (2 * n)) * np.sum(w[1:] ** 2)
        log_loss += regularization_term

        return log_loss

    def gradient(self, w):
        """Compute the gradient of the log-loss function."""
        n = len(self.y)
        predictions = self.sigmoid(self.X @ w)
        gradient = (1 / n) * self.X.T @ (predictions - self.y)

        regularization_term = (self.ridge_lambda / n) * w
        regularization_term[0] = 0
        gradient += regularization_term

        return gradient

    def gradient_descent(self, x0, lr=0.01, tol=1e-10, max_iter=50000):
        """Perform gradient descent to optimize weights."""
        xk = x0
        self.w_trace = [x0]
        k = 0

        while np.linalg.norm(self.gradient(xk)) > tol and k < max_iter:
            xk = xk - lr * self.gradient(xk)
            self.w_trace.append(xk.copy())  # Save each iteration's weights
            k += 1

        return xk

    def fit(self):
        """Fit the logistic regression model."""
        params = self.gradient_descent(self.w)
        self.w = params

    def change_threshold(self, threshold=0.5):
        self.threshold = threshold

    def predict_proba(self, X_new):
        """Predict probabilities for new data."""
        X_new = np.column_stack((np.ones(X_new.shape[0]), X_new))
        return self.sigmoid(X_new @ self.w)

    def predict(self, X_new):
        """Predict binary labels for new data."""
        probabilities = self.predict_proba(X_new)
        return (probabilities >= self.threshold).astype(int)
    
class LDA:
    def __init__(self):
        return

class MultiClassRegression: # generalizar binary para multiclase multiclass:bool=False
    def __init__(self):
        return
    
class DecisionTree:
    def __init__(self):
        return
    
class RandomForest:
    def __init__(self):
        return