## Artifitial Neural Networks

To be used both for regression and classification problems. Based on the connections between nodes, feedforward and recurrent networks as differentiated.

### Perceptron

- In the modern sense, the perceptron is an algorithm for learning a binary classifier: a function that maps its input x (a real-valued vector) to an output value f(x) (a single binary value).
- The most famous example of the perceptron's inability to solve problems with linearly nonseparable vectors is the Boolean exclusive-or problem.
- In the context of neural networks, a perceptron is an artificial neuron using the Heaviside step function as the activation function. The perceptron algorithm is also termed the single-layer perceptron, to distinguish it from a multilayer perceptron, which is a misnomer for a more complicated neural network. As a linear classifier, the single-layer perceptron is the simplest feedforward neural network.

### Cost (or loss) function

Differentiate binary and mulit-class cases for classification.

- The logarithmic loss function is the cross entropy function. Either binary_crossentropy, or categorical_crossentropy for two-class or multi-class classfication cases.
- Mean squared error
- Mean absolute error

Available loss functions in Keras: https://keras.io/losses/

### Activation functions

Differentiate binary and multi-class cases for classification. Differentiate hidden-layer and output-layer activation functions.

- Heaviside step function
- Logistic sigmoid / softmax:  the softmax function is a generalization of the logistic fuction that "squashes" a K-dimensional vector z of arbitrary real values to a K-dimensional vector sigma(z) of real values in the range 0--1 that add up to 1.
- Hyperbolic tangent
- Rectifier function = ramp function. It has been used in convolutional networks more effectively than the widely used logistic sigmoid and its more practical counterpart, the hyperbolic tangent. The rectifier is, as of 2015, the most popular activation function for deep neural networks. A unit employing the rectifier is also called a rectified linear unit (ReLU).
- Best practice: rectifier activation function for the hidden layers (found best in practice), sigmoid for the output layer activation function. Sigmoid allows for probabilistic interpretations.

Available activation functions in Keras: https://keras.io/activations/

### Weight optimization

Via some kind of (stochastic) gradient descent algorithm. Overview of algorithms (SGD: Momentum, Nesterov accelerated gradient, Adagrad, Adadelta, RMSprop, Adam, AdaMax, Nadam, as well as different algorithms to optimize asynchronous SGD) from 2016: https://arxiv.org/abs/1609.04747

- SGD
- Adam (https://arxiv.org/abs/1412.6980): Modern algorithm (~2015), uses an adaptive-learning-rate strategy. Recommended by the above paper: "Insofar, Adam might be the best overall choice." (https://arxiv.org/abs/1609.04747)

Available optimization algorithms in Keras: https://keras.io/optimizers/

### Number of layers and nodes

- Input layer: Number of nodes equals to the number of features.
- Output layer: 
    - If the NN is a regressor, then the output layer has a single node.
    - If the NN is a classifier, then it also has a single node unless softmax is used in which case the output layer has one node per class label in your model.
- Hidden layers: 
  - If data is linearly separable, no hidden layers at all are needed.
  - The situations in which performance improves with a second (or third, etc.) hidden layer are very few. One hidden layer is sufficient for the large majority of problems. Training multiple hidden layers is known to be 'hard'.
  - How many neurons? There are some empirically-derived rules-of-thumb, of these, the most commonly relied on is 'the optimal size of the hidden layer is usually between the size of the input and size of the output layers'. In practice, chosing the mean of the neurons in the input and output layers as a standard approach.

Pruning: techniques to reduce the number of 'excess' nodes (that have weights close to 0) during training to reduce computing complexity and timing.

Very brief summary by ['doug' at stackexchange](https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw)

### Code libraries

- Tensorflow
- Theano
- Keras: wrapper, capable of using both Tensorflow or Theano as backend.

### Trivia

- A single-layer (only output layer) neural network with the logistic activation function is identical to the logist regression model.
- The "Delta rule" is a gradient descent learning rule for updating the weights of the inputs to artificial neurons in a single-layer neural network. It is a special case of the more general backpropagation algorithm.


