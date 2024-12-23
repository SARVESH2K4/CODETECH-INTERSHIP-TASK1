"""
Simple logistic regression implementation.
"""
import math

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        """Compute sigmoid function."""
        return 1 / (1 + math.exp(-z))
    
    def fit(self, X, y):
        """Train the logistic regression model."""
        n_samples = len(X)
        n_features = len(X[0])
        
        # Initialize parameters
        self.weights = [0] * n_features
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.n_iterations):
            # Forward pass
            linear_preds = []
            for i in range(n_samples):
                linear_pred = sum(X[i][j] * self.weights[j] for j in range(n_features)) + self.bias
                linear_preds.append(linear_pred)
            
            predictions = [self.sigmoid(pred) for pred in linear_preds]
            
            # Backward pass
            # Update weights
            for j in range(n_features):
                dw = 0
                for i in range(n_samples):
                    dw += (predictions[i] - y[i]) * X[i][j]
                self.weights[j] -= self.learning_rate * (dw / n_samples)
            
            # Update bias
            db = sum(predictions[i] - y[i] for i in range(n_samples))
            self.bias -= self.learning_rate * (db / n_samples)
    
    def predict(self, X):
        """Make predictions on new data."""
        predictions = []
        for sample in X:
            linear_pred = sum(sample[j] * self.weights[j] for j in range(len(sample))) + self.bias
            probability = self.sigmoid(linear_pred)
            predictions.append(1 if probability >= 0.5 else 0)
        return predictions