# AI-python
simple neural net with and without tensorflow
It is a simple neural network with sigmoid function as activation function. 
it simply shows the train progress in the train()

forward propagation is responsible for X to Y,
backward propagation is for tuning the weights for each layer

this example is unable to predict any Y with value > 1 accurately, like binary coding [1 0 1] = 5 in decimal
and use it as example to train it. 

Tensorflow neural net outperforms the base_N.
It seems it has bugs in its back propagation. However, base_N is a simple NN I wrote for learning NN and testing other cases
  1. layers are extended how it affects training rate and accuracy
  2. some layers are locked (not doing back propagation) and add some layers to do, the impacts of this action.
