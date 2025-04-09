import numpy as np
import pandas as pd

class LogisticRegression:
    def __init__(self, dataset, target_name='y', ridge_lambda=0, threshold=0.5, classifier='binary', fit=True):
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
            if classifier == 'binary':
                self.fit_binary()
            elif classifier == 'multi-class':
                self.fit_multiclass()
            else:
                raise Exception("Invalid classifier type. Can only accept 'binary' or 'multi-class'")

    

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
    