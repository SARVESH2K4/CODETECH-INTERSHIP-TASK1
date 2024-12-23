"""
Training script for the CNN model.
"""
from model import SimpleCNN
from data import load_sample_data

def main():
    # Load sample data
    print("Loading sample data...")
    image, label = load_sample_data()
    
    # Initialize model
    print("\nInitializing CNN model...")
    model = SimpleCNN(input_shape=(8, 8), num_classes=10)
    
    # Forward pass
    print("\nPerforming forward pass...")
    output = model.forward(image)
    
    # Print results
    print("\nModel output probabilities:")
    for i, prob in enumerate(output):
        print(f"Class {i}: {prob:.4f}")

if __name__ == "__main__":
    main()