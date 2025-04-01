import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, dataset, target_name='y', fit=True):
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

        if fit:
            self.fit()

    def sigmoid(self, z):
        """Apply the sigmoid function."""
        return 1 / (1 + np.exp(-z))

    def gradient(self, w):
        """Compute the gradient of the log-loss function."""
        n = len(self.y)
        predictions = self.sigmoid(self.X @ w)
        return (1 / n) * self.X.T @ (predictions - self.y)

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

    def predict_proba(self, X_new):
        """Predict probabilities for new data."""
        X_new = np.column_stack((np.ones(X_new.shape[0]), X_new))
        return self.sigmoid(X_new @ self.w)

    def predict(self, X_new, threshold=0.5):
        """Predict binary labels for new data."""
        probabilities = self.predict_proba(X_new)
        return (probabilities >= threshold).astype(int)
    
def main():
    data = pd.read_csv('datasets/horas_de_estudio.csv')
    model = LogisticRegression(dataset=data, target_name='Aprobado')
    pred = model.predict(np.array([3.6]))
    print(pred)

    plt.scatter(data['Horas_Estudiadas'], data['Aprobado'], c=data['Aprobado'], cmap='bwr', label='Data points')
    
    # Calculate the decision boundary
    x_values = np.linspace(data['Horas_Estudiadas'].min(), data['Horas_Estudiadas'].max(), 100)
    w = model.w  # Get the weights from the model

if __name__ == '__main__':
    main()