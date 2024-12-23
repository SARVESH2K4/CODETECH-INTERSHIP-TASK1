"""
Main script to run the classification experiment.
"""
from data import generate_dataset
from model import LogisticRegression
from utils import train_test_split, evaluate_model

def main():
    # Generate synthetic dataset
    print("Generating dataset...")
    X, y = generate_dataset(n_samples=1000)
    
    # Split data
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Train model
    print("\nTraining logistic regression model...")
    model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
    model.fit(X_train, y_train)
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred = model.predict(X_test)
    
    # Evaluate model
    print("\nEvaluating model performance...")
    metrics = evaluate_model(y_test, y_pred)
    
    # Print results
    print("\nModel Performance Metrics:")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")

if __name__ == "__main__":
    main()