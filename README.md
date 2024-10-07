#4-Layer Neural Network

This module implements a 4-layer neural network that uses backpropagation to learn a mapping from a binary input dataset to a binary output dataset. The network is designed to train and predict binary outputs based on a 4x4 input matrix.

Table of Contents

	•	Overview
	•	Network Architecture
	•	Activation Function
	•	Training Process
	•	Testing
	•	How to Run
	•	Dependencies

Overview

This neural network takes a binary input dataset and trains itself to predict the binary output using the backpropagation algorithm. It uses a sigmoid activation function to model non-linearity and gradient descent to minimize error.

The purpose of this network is to demonstrate a basic implementation of a feedforward neural network with backpropagation.

Network Architecture

The network consists of four layers:

	1.	Input Layer (l0): A 4x4 matrix representing the input data points.
	2.	First Hidden Layer (l1): A layer of 4 neurons.
	3.	Second Hidden Layer (l2): Another layer of 4 neurons.
	4.	Output Layer (l3): A single neuron that produces a binary output.

Synaptic weights between each layer are randomly initialized and updated during training.

Variables:

	•	syn0: Weights between the input layer (l0) and the first hidden layer (l1).
	•	syn1: Weights between the first hidden layer (l1) and the second hidden layer (l2).
	•	syn2: Weights between the second hidden layer (l2) and the output layer (l3).

Activation Function

The network uses the sigmoid activation function:

s(x) = 1 / (1 + e^(-cx))

	•	The sigmoid function is used for non-linearity and to ensure smooth gradient updates.
	•	Its derivative is used for calculating the weight adjustments during backpropagation:

s'(x) = s(x) * (1 - s(x))

Training Process

The network is trained using backpropagation, which minimizes the error between the actual output (y) and the predicted output (l3). The error is propagated backward through the layers to adjust the weights.

Forward Propagation:

	•	Each layer is calculated by performing a dot product between the previous layer and the corresponding synaptic weights.
	•	The result is passed through the sigmoid activation function to get the next layer’s values.

Backpropagation:

	1.	The error for the output layer (l3_error) is calculated as the difference between the actual output and the predicted output.
	2.	The error is propagated backward through the network using the derivative of the sigmoid function to update the weights in each layer.

Testing

The testSweep() function allows you to test the network with all possible binary inputs (16 combinations of 4 bits) to evaluate its performance after training. The test() function can also be used to test the network with specific input combinations.

How to Run

	1.	Install Dependencies: Ensure that you have Python and NumPy installed.
	2.	Run the Code: Simply execute the script in your terminal or Python environment.

python nnet4.py

The network will train over 80,000 iterations, printing the error and weight updates every 10,000 iterations.

Sample Output

 input       | output
[0, 0, 0, 0] | [[0.00212042]]
[0, 0, 0, 1] | [[0.00383463]]
[0, 0, 1, 0] | [[0.99781954]]
...

Dependencies

	•	Python 3.x
	•	NumPy
	•	Install via pip if not already installed:

pip install numpy

This simple neural network is a starting point for understanding how feedforward networks work, particularly how backpropagation is used to train neural networks. You can modify the code to experiment with different architectures, activation functions, or datasets.
