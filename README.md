# DeepLearning
In this repository, there are 3 projects. Sentiment analysis with CNN's, sentiment analysis with RNN's and a simple Deep Q-learning algorithm for the game 'LunarLander-v2'



For the first project, using kernel matrices for the convolution, we can extract features of our sentences. Those features are called n-grams (the n parameter is about the filter size) and play an important role in the sentiment analysis. Then, adding a max pool layer, we can extract the most important feature (word) of our sentece & reduce the dimensionality. Dropout is also being implemented, as a regularization method.



For the second project, I used a bidirectional, 2-layer LSTM. Each sentece can be seen a sequence of words, so the LSTM is an ideal architecutre for the sentiment analysis. The output of first LSTM is the input to the second LSTM. Each of those are divided into forward and backward LSTM's (they read the sentences in two different directions). The prediction uses the forward and backward hidden states of the second LSTM.



For the third project, a simple fully connected network (activated with ReLU) is used as the network that will estimate the Q-learning matrix. The last layer does not have an activation function, in order to get the raw estimates of the q-values. Then, a replay memory is constructed in order to select random batches for the learning procedure. Also, the exploitation - exploration dilemma is being handled by initializing epsilon at the value 1 and then reducing it in a linear way. 
