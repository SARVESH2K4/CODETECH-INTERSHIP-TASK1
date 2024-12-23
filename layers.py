"""
Neural network layer implementations.
"""
import math
import random

class ConvLayer:
    def __init__(self, in_channels, out_channels, kernel_size):
        self.kernel_size = kernel_size
        scale = math.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.filters = [
            [[random.gauss(0, scale) for _ in range(kernel_size)]
             for _ in range(kernel_size)]
            for _ in range(out_channels)
        ]
        self.biases = [0] * out_channels
    
    def forward(self, input_data):
        """Apply convolution operation."""
        height, width = len(input_data), len(input_data[0])
        output_height = height - self.kernel_size + 1
        output_width = width - self.kernel_size + 1
        
        output = [[[0] * output_width for _ in range(output_height)]
                 for _ in range(len(self.filters))]
        
        for f_idx, filter_kernel in enumerate(self.filters):
            for i in range(output_height):
                for j in range(output_width):
                    # Compute convolution at this position
                    sum_val = 0
                    for ki in range(self.kernel_size):
                        for kj in range(self.kernel_size):
                            sum_val += (input_data[i + ki][j + kj] * 
                                      filter_kernel[ki][kj])
                    output[f_idx][i][j] = sum_val + self.biases[f_idx]
        
        return output

class MaxPoolLayer:
    def __init__(self, pool_size=2):
        self.pool_size = pool_size
    
    def forward(self, input_data):
        """Apply max pooling operation."""
        channels = len(input_data)
        height = len(input_data[0])
        width = len(input_data[0][0])
        
        output_height = height // self.pool_size
        output_width = width // self.pool_size
        
        output = [[[0] * output_width for _ in range(output_height)]
                 for _ in range(channels)]
        
        for c in range(channels):
            for i in range(output_height):
                for j in range(output_width):
                    # Find maximum in the pooling window
                    max_val = float('-inf')
                    for pi in range(self.pool_size):
                        for pj in range(self.pool_size):
                            val = input_data[c][i*self.pool_size + pi][j*self.pool_size + pj]
                            max_val = max(max_val, val)
                    output[c][i][j] = max_val
        
        return output