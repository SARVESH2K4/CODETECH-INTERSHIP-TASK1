# CODETECH-INTERSHIP-TASK1
# Simple CNN Implementation in Python

A lightweight implementation of a Convolutional Neural Network (CNN) from scratch using pure Python. This project demonstrates core deep learning concepts without external dependencies.

## Project Structure

```
├── activation.py        # Activation functions (ReLU, Softmax)
├── data.py             # Data loading and preprocessing utilities
├── layers/             # Neural network layer implementations
│   ├── __init__.py
│   ├── base.py        # Base layer class
│   ├── conv.py        # Convolutional layer
│   └── pooling.py     # Max pooling layer
├── model.py           # CNN model implementation
├── train.py          # Training script
└── utils/            # Utility functions
    ├── __init__.py
    └── tensor.py     # Tensor manipulation utilities
```

## Features

- Modular CNN implementation
- Convolutional and max pooling layers
- ReLU and Softmax activation functions
- Basic tensor operations
- Sample data loader for testing

## Usage

```python
from model import SimpleCNN
from data import load_sample_data

# Load sample data
image, label = load_sample_data()

# Initialize model
model = SimpleCNN(input_shape=(8, 8), num_classes=10)

# Forward pass
output = model.forward(image)
```

## Implementation Details

### Layers
- **ConvLayer**: Implements 2D convolution operation
- **MaxPoolLayer**: Implements 2x2 max pooling
- **Fully Connected Layer**: Final classification layer

### Activation Functions
- ReLU: `f(x) = max(0, x)`
- Softmax: For output probability distribution

## Contributing

Feel free to open issues or submit pull requests for improvements.

## License

MIT License
