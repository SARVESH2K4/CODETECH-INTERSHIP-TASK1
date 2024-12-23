"""
Generates a simple synthetic dataset for binary classification.
"""
import random

def generate_dataset(n_samples=100):
    """Generate a synthetic dataset for binary classification."""
    X = []  # Features
    y = []  # Labels
    
    # Generate two clusters of points
    for _ in range(n_samples):
        # Class 0: Points centered around (1, 1)
        if random.random() > 0.5:
            x1 = random.gauss(1, 0.5)
            x2 = random.gauss(1, 0.5)
            X.append([x1, x2])
            y.append(0)
        # Class 1: Points centered around (4, 4)
        else:
            x1 = random.gauss(4, 0.5)
            x2 = random.gauss(4, 0.5)
            X.append([x1, x2])
            y.append(1)
    
    return X, y