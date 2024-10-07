# Four-Layer Neural Network

This project implements a four-layer neural network that uses backpropagation to learn a function mapping input data to output data. The network is designed to handle binary input (4x4 matrix) and binary output (4x1 matrix). The neural network adjusts synaptic weights through training to minimize error and achieve accurate predictions.

## Features
- **Four Layers**: Includes an input layer, two hidden layers, and an output layer.
- **Backpropagation**: Minimizes error using gradient descent and adjusts weights accordingly.
- **Sigmoid Activation Function**: A smooth, differentiable function used to determine neuron activation.
- **Testing**: Includes a sweep of test cases to validate the learned model after training.

## Getting Started

### Prerequisites
- Python 3.x
- NumPy
- Six module (for compatibility)

You can install the required dependencies using:
```bash
pip install numpy six
```

### Running the Code
The main class in the code is `nnet4`, which implements the neural network. The following key functions are included:

- **`train()`**: Trains the neural network using a dataset and adjusts synaptic weights to minimize error.
- **`test()`**: Tests the network with a given input to see how well it predicts the output.
- **`testSweep()`**: Tests the network across all possible binary inputs (not seen during training) to evaluate generalization.

### To run the training and testing process:
```python
from nnet4 import nnet4

nn = nnet4()
nn.run()  # Trains the network and performs a test sweep
```

## Code Overview

### Neural Network Layers
- **Input Layer (l0)**: Represents the input dataset, a matrix of binary data.
- **Hidden Layers (l1, l2)**: Intermediate layers between input and output, learning abstract representations of the input.
- **Output Layer (l3)**: The predicted output, compared with the expected values during training.

### Synaptic Weights
- **`syn0`**: Weights connecting the input layer (l0) to the first hidden layer (l1).
- **`syn1`**: Weights connecting the first hidden layer (l1) to the second hidden layer (l2).
- **`syn2`**: Weights connecting the second hidden layer (l2) to the output layer (l3).

### Training Process
1. **Forward Propagation**: Each layer is computed using the dot product of the previous layer and the associated weights, followed by applying the sigmoid activation function.
2. **Error Calculation**: The error between the predicted and actual output is calculated.
3. **Backpropagation**: Gradients of the error are computed, and weights are updated accordingly using the delta rule and gradient descent.

### Functions
- **`nonlin(x, c=1, deriv=False)`**: The sigmoid activation function, with an optional derivative mode.
- **`train(iter=80000)`**: Trains the network for a specified number of iterations, adjusting the synaptic weights to minimize error.
- **`test(inp=[0,0,0,0])`**: Tests the current synaptic weights using the specified input.
- **`testSweep()`**: Runs the test function on all possible binary inputs to evaluate the network's performance.

## Notes
- The network uses the sigmoid function as the activation function, which ensures continuity and differentiability, essential for backpropagation.
- The dataset is hardcoded within the `nnet4` class, consisting of 5 training examples.
- The error and weight updates are displayed periodically during training for monitoring progress.

## License
This project is open-source and available under the MIT License.
