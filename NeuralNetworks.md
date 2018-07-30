# Artificial Neural Networks

To be used both for regression and classification problems. Based on the connections between nodes, feedforward and recurrent networks are differentiated.

## Free books

- Michael Nielsen's free introductory book "[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com)".<br>
- Ian Goodfellow, Yoshua Bengio, and Aaron Courville's "[Deep Learning](http://www.deeplearningbook.org/)", [lectures](http://www.deeplearningbook.org/lecture_slides.html) are also available.<br>
- David Kriesel's "[A Brief Introduction to Neural Networks](http://www.dkriesel.com/en/science/neural_networks)".<br>
- Martin T. Hagan, Howard B. Demuth, Mark H. Beale and Orlando D. Jess on "[Neural Network Design](http://hagan.ecen.ceat.okstate.edu/nnd.html)".<br>
- Simon Haykin on "[Neural Networks and Learning Machines](https://cours.etsmtl.ca/sys843/REFS/Books/ebook_Haykin09.pdf)".<br>
- Li Deng and Dong Yu's "[Deep Learning Methods and Applications](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/DeepLearning-NowPublishing-Vol7-SIG-039.pdf)" on signal and information processing tasks.<br>
- Python code recipes by LISA Lab, University of Montreal, "[Deep Learning Tutorial](http://deeplearning.net/tutorial/deeplearning.pdf)" using Theano.<br>

## Perceptron

- In the modern sense, the perceptron is an algorithm for learning a binary classifier: a function that maps its input x (a real-valued vector) to an output value f(x) (a single binary value).
- The most famous example of the perceptron's inability to solve problems with linearly nonseparable vectors is the Boolean exclusive-or problem.
- In the context of neural networks, a perceptron is an artificial neuron using the Heaviside step function as the activation function. The perceptron algorithm is also termed the single-layer perceptron, to distinguish it from a multilayer perceptron, which is a misnomer for a more complicated neural network. As a linear classifier, the single-layer perceptron is the simplest feedforward neural network.

## Activation functions

Differentiate binary and multi-class cases for classification. Differentiate hidden-layer and output-layer activation functions.
The activation function at the output layer often depends on the cost function.

Hidden layer:
- Heaviside step function
- Rectifier function = ramp function. It has been used in convolutional networks more effectively than the widely used logistic sigmoid and its more practical counterpart, the hyperbolic tangent. The rectifier is, as of 2015, the most popular activation function for deep neural networks. A unit employing the rectifier is also called a rectified linear unit (ReLU).
- Hyperbolic tangent: suffers from the vanishing gradients problem
- Sigmoid: suffers from the vanishing gradients problem
- Best practice: rectifier activation function for the hidden layers
 
Output layer:
- Linear: for regression (using simoid, for instance, for regression would limit the predictions between the supplied minimum and maximum: sigmoid is between 0 and 1).
- Logistic sigmoid / softmax: for classification; the softmax function is a generalization of the logistic fuction that "squashes" a K-dimensional vector z of arbitrary real values to a K-dimensional vector sigma(z) of real values in the range 0--1 that add up to 1.
- (The output of convolutional networks could be ReLU.)

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

- For classification:
    - The logarithmic loss function is the cross entropy function (either binary_crossentropy, or categorical_crossentropy for two-class or multi-class classfication cases), which is tipically used for softmax layers. The cross entropy loss is ubiquitous in modern deep neural networks.
    - Hinge loss
    - Logistic loss
    - Kullback–Leibler divergence
    - Exponential loss
- For regression (linear layers):
    - Tipical choice is mean squared error: it has the disadvantage that it has the tendency to be dominated by outliers
    - Mean absolute error or Laplace or L1 loss: better than MSE in case of outliers, however, it is not differentiable at 0. One problem with using MAE for training of neural nets is its constantly large gradient, which can lead to missing minima at the end of training using gradient descent.
    - Huber loss: Huber loss is less sensitive to outliers in data than the squared error loss. It’s also differentiable at 0. Deoending on the hyperparameter delta, the Huber loss approaches MAE when delta approaches 0, and MSE when delta approaches infinity. Huber loss combines good properties from both MSE and MAE: less sensitivity to outliers than MSE, but changing gradient around the minima unlike in MAE. HThe problem with Huber loss is that one might need to train the hyperparameter delta.
    - Log cosh loss: Log-cosh is the logarithm of the hyperbolic cosine of the prediction error. Log cosh is approximately equal to (x^2)/2 for small x and to abs(x)-log(2) for large x. This means that log cosh works mostly like the mean squared error, but will not be so strongly affected by the occasional wildly incorrect prediction. It has all the advantages of Huber loss, and it’s twice differentiable everywhere, unlike Huber loss. Having second order derivatives exist everywhere (Hessian matrix) might be matter for some optimization algorithms. Log cosh still suffers from the problem of gradient and hessian for very large off-target predictions being constant, therefore resulting in the absence of splits for XGBoost.
    - Quantile loss: Quantile loss functions are useful when we are interested in predicting an interval instead of only point predictions. Quantile-based regression aims to estimate the conditional "quantile" of a response variable given certain values of predictor variables. Quantile loss is an extension of MAE: when quantile is 50th percentile, it is MAE. The idea is to choose the quantile value based on whether we want to give more value to positive errors or negative errors. Loss function tries to give different penalties to overestimation and underestimation based on the value of chosen quantile.

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
    - Examples would include Newton's method (which is applied to f' instead of f, and so it gets the Hessian involved), and the interior-point algorithms https://web.stanford.edu/class/msande311/lecture13.pdf

Selected algorithms:
- Newton's method: second order algorithm, where Newton's root searching algorithm is applied to f' (and thus it involes the Hessian). Saddle points trap this kind of optimization, however linesearch with the Wolfe conditions or using or trust regions prevents convergence to saddle points.
- Gradient descent: minimize or maximize a loss function E(x) using its gradient values with respect to the parameters. It is a first order optimization algorithm, that is to say that it only uses the first order derivatives (the Jacobian matrix).
- Stochastic gradient descent: frequent updates to the weights (compared to GD), which results in higher variance in the loss function, which in turn helps to discover 'better' minima. The stochastic nature will result in more complex convergence to the exact minima. 
- Mini batch gradient descent: updates are only performed after batches of inputs have been processed. In practice, SGD usually refers to this batch algorithm.
- Momentum GD: improves convergence by making the amout of the update of weigths depend on the update in the last step. Thus, if derivatives consistently point toward the same direction in each step, the update gets larger, if the derivatives change direction, the update slows down. This means it does parameter updates only for relevant examples. This reduces the unnecessary parameter updates which leads to faster and stable convergence and reduced oscillations.
- Nesterov accelerated gradient: an improvement over Momentum GD: it would be great if the step size of the parameter update would somehow know that we start to approach the minimum and thus would start to decrease before reaching the minimum. The NAG algorithm looks ahead to the anticipated parameter pozition in the future, and it adds a correction to the parameter update step based on the derivative at the future position. 
- Adagrad: Previously, all parameters used the same learning rate (which depended on the time step for Momentum GD and NAG). Adagrad uses a different learning rate for every parameter at every time step t.
- AdaDelta: It is an extension of AdaGrad which tends to remove the decaying learning Rate problem of it. Instead of accumulating all previous squared gradients, Adadelta limits the window of accumulated past gradients to some fixed size w.
- Adam = Adaptive Moment Estimation (https://arxiv.org/abs/1412.6980): In addition to storing an exponentially decaying average of past squared gradients like AdaDelta ,Adam also keeps an exponentially decaying average of past gradients M(t), similar to momentum. Modern algorithm (~2015), uses an adaptive-learning-rate strategy. Recommended by the above paper: "Insofar, Adam might be the best overall choice." (https://arxiv.org/abs/1609.04747)

The standard Newton's method is trapped by saddle points. Being trapped at saddle points _might_ be possible in gradient descent as well (even though it is very unlikely with random initialization). In momentum-using methods, like AdaGrad or Adam, saddle points are surely escaped as the past gradients are also used.

The above summary is mostly from a [towardsdatascience.com article](https://towardsdatascience.com/types-of-optimization-algorithms-used-in-neural-networks-and-ways-to-optimize-gradient-95ae5d39529f)<br>
Available optimization algorithms in Keras: https://keras.io/optimizers/<br>
Computation of backpropagation formulae for common neural networks: https://www.ics.uci.edu/~pjsadows/notes.pdf

## Number of layers and nodes

- Input layer: Number of nodes equals to the number of features.
- Output layer: 
    - If the NN is a regressor, then the output layer has a single node.
    - If the NN is a classifier, then it also has a single node unless softmax is used in which case the output layer has one node per class label in your model.
- Hidden layers: 
  - If data is linearly separable, no hidden layers at all are needed.
  - The situations in which performance improves with a second (or third, etc.) hidden layer are very few. One hidden layer is sufficient for the large majority of problems. Training multiple hidden layers is known to be 'hard'.
  - Lots of noise, little structure -> not deep
  - Little noise, complex structure -> deep (larger representational capacity)
  - How many neurons? There are some empirically-derived rules-of-thumb, of these, the most commonly relied on is 'the optimal size of the hidden layer is usually between the size of the input and size of the output layers'. In practice, chosing the mean of the neurons in the input and output layers as a standard approach.

Pruning: techniques to reduce the number of 'excess' nodes (that have weights close to 0) during training to reduce computing complexity and timing.

Very brief summary by ['doug' at stackexchange](https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw)

## Regularization (prevent overfitting)

There are three broad ways of preventing overfitting: i) constraining the parameters, ii) ensmeble of many models, and iii) more training data.

- Classical L1 and L2 parameter norm penalties (to the cost function): The L1 regularization has the intriguing property that it leads the weight vectors to become sparse during optimization (i.e. very close to exactly zero). In other words, neurons with L1 regularization end up using only a sparse subset of their most important inputs and become nearly invariant to the 'noisy' inputs. In comparison, final weight vectors from L2 regularization are usually diffuse, small numbers. In practice, if you are not concerned with explicit feature selection, L2 regularization can be expected to give superior performance over L1.
- Early stopping: Early stopping rules provide guidance as to how many iterations can be run before the learner begins to over-fit. 
- Parameter tying and parameter sharing
- Bagging and other ensemble methods
- Droupout
Dropout is a regularization technique for neural network models proposed by Srivastava, et al. in their 2014 [paper](http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)
It is a technique where randomly selected neurons are ignored during training. They are “dropped-out” randomly. This means that their contribution to the activation of downstream neurons is temporally removed on the forward pass and any weight updates are not applied to the neuron on the backward pass.
As a neural network learns, neuron weights settle into their context within the network. Weights of neurons are tuned for specific features providing some specialization. Neighboring neurons become to rely on this specialization, which if taken too far can result in a fragile model too specialized to the training data. This reliance on context for a neuron during training is referred to as complex co-adaptation. Droupout does not modify the cost function. Drouput can be view as an ensemble method as it averages the weights computed across all the modified networks. See [Jason Brownlee's tips](https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/) for using Dropout.
- Data augumentation
- Tangent propagation
- Batch normalization also sometimes reduces generalization error and allows dropout to be omitted, because of the noise in the estimate of the statistics used to normalize each variable.

Parameter norm regularizers in Keras: https://keras.io/regularizers/<br>
Dropout in Keras: https://keras.io/layers/core/#dropout

## Code libraries

- Tensorflow
- Theano
- Keras: wrapper, capable of using both Tensorflow or Theano as backend.

## Trivia

- A single-layer (only output layer) neural network with the logistic activation function is identical to the logist regression model.
- The "Delta rule" is a gradient descent learning rule for updating the weights of the inputs to artificial neurons in a single-layer neural network. It is a special case of the more general backpropagation algorithm.

## Why do artificial neural networks 'work'?

This question will be broken down into its implicit subquestions below.

### What kind of functions can artifitial neural networks model?

Ultimately, each network describes a rule for transforming input into output: each network is an 'embodiment' of a function. The nodes / blocks of the network are subfunctions. The structure of the network determines in which order the various subfunctions are evaluated. 

What kind of functions can be represented by the network? This is a mathematical question with known answers under certain conditions. See the [Universal approximation theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem). The theorem states that a feed-forward network with a single hidden layer containing a finite number of neurons can approximate continuous functions on compact subsets of R^n, under mild assumptions on the activation function. 

Formally, by [George Cybenko for the sigmoid activation function](https://doi.org/10.1007/BF02551274):<br>
Let phi be a nonconstant, bounded, and monotonically-increasing continuous function. Let I_m denote the m-dimensional unit hypercube \[0,1\]^m. The space of continuous functions on I_m is denoted by C(I_m). Then, given any epsilon > 0 and any function f in C(I_m), there exist an integer N, real constants v_i , b_i in R and real vectors w_i in R^m, where i = 1 , ..., N , such that we may define:<br> 
F(x) = Sum_{i = 1}^{N} (v_i phi(w_i^T * x + b_i)) such that |F(x) - f(x)| < epsilon for all x in I_m.<br>
The statememt even holds if I_m is replaced by any compact subset of R^m.

For ANN, we can recognize that:
- phi: activation function (sigmoid is noncontant, bounded, monotonically-increasing, continuous)
- w_i: weigths in the input vector for the i^th neuron
- b_i: bias on the i^th neuron
- I_m: represents feature normalization, however the theorem holds with compact subsets of R^m too
- Summation: output node activation after one hidden layer of N neurons with linear activation function

[Kurt Hornik showed](https://doi.org/10.1016/0893-6080(91)90009-T) in 1991 that it is not the specific choice of the activation function, but rather the multilayer feedforward architecture itself which gives neural networks the potential of being universal approximators.




