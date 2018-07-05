# Artifitial Neural Networks

Michael Nielsen's free introductory book: http://neuralnetworksanddeeplearning.com
Ian Goodfellow, Yoshua Bengio, and Aaron Courville's
- free book: http://www.deeplearningbook.org/
- lectures: http://www.deeplearningbook.org/lecture_slides.html

To be used both for regression and classification problems. Based on the connections between nodes, feedforward and recurrent networks as differentiated.

## Perceptron

- In the modern sense, the perceptron is an algorithm for learning a binary classifier: a function that maps its input x (a real-valued vector) to an output value f(x) (a single binary value).
- The most famous example of the perceptron's inability to solve problems with linearly nonseparable vectors is the Boolean exclusive-or problem.
- In the context of neural networks, a perceptron is an artificial neuron using the Heaviside step function as the activation function. The perceptron algorithm is also termed the single-layer perceptron, to distinguish it from a multilayer perceptron, which is a misnomer for a more complicated neural network. As a linear classifier, the single-layer perceptron is the simplest feedforward neural network.

## Activation functions

Differentiate binary and multi-class cases for classification. Differentiate hidden-layer and output-layer activation functions.

- Heaviside step function
- Logistic sigmoid / softmax:  the softmax function is a generalization of the logistic fuction that "squashes" a K-dimensional vector z of arbitrary real values to a K-dimensional vector sigma(z) of real values in the range 0--1 that add up to 1.
- Hyperbolic tangent
- Rectifier function = ramp function. It has been used in convolutional networks more effectively than the widely used logistic sigmoid and its more practical counterpart, the hyperbolic tangent. The rectifier is, as of 2015, the most popular activation function for deep neural networks. A unit employing the rectifier is also called a rectified linear unit (ReLU).
- Best practice: rectifier activation function for the hidden layers (found best in practice), sigmoid for the output layer activation function. Sigmoid allows for probabilistic interpretations.

Available activation functions in Keras: https://keras.io/activations/

### ReLU

Advantages:
- Its output is a true zero (not just a small value close to zero) for z <= 1
- Its derivative is constant, either 0 for z < 0 or 1 for z > 0. 
- Biological plausibility: One-sided, compared to the antisymmetry of tanh.
- Sparse activation: For example, in a randomly initialized network, only about 50% of hidden units are activated (having a non-zero output).
- Better gradient propagation: Fewer vanishing gradient problems compared to sigmoidal activation functions that saturate in both directions.
- Efficient computation: Only comparison, addition and multiplication.
- Scale-invariant: max(0, ax) = a max(0,x) for a >= 0.

Potential problem:
- Derivatives at x = 0 can be dealt with by
    - Assign arbitrary values; common values are 0, 0.5, and 1.
    - Instead of using the actual y = ReLU(x) function, use an approximation to ReLU which is differentiable for all values of x. One such approximation is called softplus which is defined y = ln(1.0 + e^x) which has derivative of y’ = 1.0 / (1.0 + e^-x) which is, remarkably, the logistic sigmoid function! This would, however, seem to nullify both advantages mentioned above...
- Non-zero centered
- Unbounded
- Dying ReLU problem: ReLU neurons can sometimes be pushed into states in which they become inactive for essentially all inputs. In this state, no gradients flow backward through the neuron, and so the neuron becomes stuck in a perpetually inactive state and "dies." This is a form of the "vanishing gradient problem." In some cases, large numbers of neurons in a network can become stuck in dead states, effectively decreasing the model capacity. This problem typically arises when the learning rate is set too high. It may be mitigated by using Leaky ReLUs instead.

## Cost (or loss) function

There are some criteria to keep in mind when defining cost functions:
- One of the most important ones is to make the C cost function differentiable with respect to all the outputs (the y’s). This is necessary for gradient descent to work.
- Another good constraint is to make the cost of many inputs the average of the cost of individual inputs.

- For classification: The logarithmic loss function is the cross entropy function (either binary_crossentropy, or categorical_crossentropy for two-class or multi-class classfication cases), which is tipically used for softmax layers.
- For regression (linear layers): Tipical choice is mean squared error
- Mean absolute error: however, it is not differentiable at 0.

Available loss functions in Keras: https://keras.io/losses/

## Weight optimization

Neural networks trains via a famous technique called Backpropagation, in which we first propagate forward calculating the dot product of inputs signals and their corresponding eeights, and then apply an activation function to those sum of products, which transforms the input signal to an output signal (and also is important to model complex non-linear functions and introduces non-linearities to the model, which enables the model to learn almost any arbitrary functional mappings).

After this we propagate backwards in the network carrying error terms and updating weights values using gradient descent, in which we calculate the gradient of Error(E) function with respect to the weights or the parameters, and update the weights or parameters in the opposite direction of the gradient of the loss function.

Overview of algorithms (SGD: Momentum, Nesterov accelerated gradient, Adagrad, Adadelta, RMSprop, Adam, AdaMax, Nadam, as well as different algorithms to optimize asynchronous SGD) from 2016: https://arxiv.org/abs/1609.04747

Broad categories:
- First order algorithms: they only use the first order derivatives (the Jacobian matrix). They give a line that is tangential to a point on its Error Surface.
    - Examples would include the gradient descent algoritmhs
- Second order algoritmhs: they use the second order derivatives (the Hessian matrix). They provide us with a quadratic surface which touches the curvature of the Error Surface.
    - Such algirithms are costly to compute compared to the first order ones, however they will not get stuck around paths of slow convergence around saddle points whereas gradient descent sometimes gets stuck and does not converges.
    - Examples would include the interior-point algorithms https://web.stanford.edu/class/msande311/lecture13.pdf

Selected algorithms:
- Gradient descent: minimize or maximize a loss function E(x) using its gradient values with respect to the parameters. It is a first order optimization algorithm, that is to say that it only uses the first order derivatives (the Jacobian matrix).
- Stochastic gradient descent: frequent updates to the weights (compared to GD), which results in higher variance in the loss function, which in turn helps to discover 'better' minima. The stochastic nature will result in more complex convergence to the exact minima. 
- Mini batch gradient descent: updates are only performed after batches of inputs have been processed. In practice, SGD usually refers to this batch algorithm.
- Momentum GD: improves convergence by making the amout of the update of weigths depend on the update in the last step. Thus, if derivatives consistently point toward the same direction in each step, the update gets larger, if the derivatives change direction, the update slows down. This means it does parameter updates only for relevant examples. This reduces the unnecessary parameter updates which leads to faster and stable convergence and reduced oscillations.
- Nesterov accelerated gradient: an improvement over Momentum GD: it would be great if the step size of the parameter update would somehow know that we start to approach the minimum and thus would start to decrease before reaching the minimum. The NAG algorithm looks ahead to the anticipated parameter pozition in the future, and it adds a correction to the parameter update step based on the derivative at the future position. 
- Adagrad: Previously, all parameters used the same learning rate (which depended on the time step for Momentum GD and NAG). Adagrad uses a different learning rate for every parameter at every time step t.
- AdaDelta: It is an extension of AdaGrad which tends to remove the decaying learning Rate problem of it. Instead of accumulating all previous squared gradients, Adadelta limits the window of accumulated past gradients to some fixed size w.
- Adam = Adaptive Moment Estimation (https://arxiv.org/abs/1412.6980): In addition to storing an exponentially decaying average of past squared gradients like AdaDelta ,Adam also keeps an exponentially decaying average of past gradients M(t), similar to momentum. Modern algorithm (~2015), uses an adaptive-learning-rate strategy. Recommended by the above paper: "Insofar, Adam might be the best overall choice." (https://arxiv.org/abs/1609.04747)

The above summary is mostly from a [towardsdatascience.com article](https://towardsdatascience.com/types-of-optimization-algorithms-used-in-neural-networks-and-ways-to-optimize-gradient-95ae5d39529f)
Available optimization algorithms in Keras: https://keras.io/optimizers/
Computation of backpropagation formulae for common neural networks: https://www.ics.uci.edu/~pjsadows/notes.pdf

## Number of layers and nodes

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

## Code libraries

- Tensorflow
- Theano
- Keras: wrapper, capable of using both Tensorflow or Theano as backend.

## Trivia

- A single-layer (only output layer) neural network with the logistic activation function is identical to the logist regression model.
- The "Delta rule" is a gradient descent learning rule for updating the weights of the inputs to artificial neurons in a single-layer neural network. It is a special case of the more general backpropagation algorithm.


