import sys
import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).
  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.
  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength
  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] += X[i].T 
        dW[:,y[i]] -= X[i].T

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.
  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  # pass
  # print('[DEBUG] X.shape = {}'.format(X.shape))
  num_train = X.shape[0]
  S = X.dot(W)
  # print('[DEBUG] S type: {}'.format(S.dtype))
  # print('[DEBUG] y type: {}'.format(y.dtype))
  # print('[DEBUG] num train',num_train)
  # print(np.choose(y, S.T))
  correct_class_score = np.choose(y, S.T).T.reshape(num_train,1) 

  margins = np.maximum(0, S - correct_class_score + 1) 
  margins[np.arange(margins.shape[0]), y] = 0

  loss = np.sum(margins, axis = (0, 1))# - num_train
  loss /= num_train + 0.5 * reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  # pass
  F = (margins > 0).astype(int)
  F[np.arange(F.shape[0]),y] = np.sum(-F, axis = 1)
  dW = X.T.dot(F)/num_train + reg*W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW


class LinearClassifier(object):

    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
                batch_size=64, verbose=False):
        """
        Train this linear classifier using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
        training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
        means that X[i] has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        num_train, dim = X.shape
        num_classes = np.max([int(label) for label in y]) + 1 # assume y takes values 0...K-1 where K is number of classes
        if self.W is None:
        # lazily initialize W
            self.W = 0.001 * np.random.randn(dim, num_classes)

        # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None

        #########################################################################
        # TODO:                                                                 #
        # Sample batch_size elements from the training data and their           #
        # corresponding labels to use in this round of gradient descent.        #
        # Store the data in X_batch and their corresponding labels in           #
        # y_batch; after sampling X_batch should have shape (dim, batch_size)   #
        # and y_batch should have shape (batch_size,)                           #
        #                                                                       #
        # Hint: Use np.random.choice to generate indices. Sampling with         #
        # replacement is faster than sampling without replacement.              #
        #########################################################################
        # pass
        idxs = np.random.choice(num_train, batch_size)
        X_batch = X[idxs, :]
        y_batch = y[idxs]
        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################

        # evaluate loss and gradient
        loss, grad = self.loss(X_batch, y_batch, reg)
        loss_history.append(loss)

        # perform parameter update
        #########################################################################
        # TODO:                                                                 #
        # Update the weights using the gradient and the learning rate.          #
        #########################################################################
        # pass
        self.W -= learning_rate*grad
        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################

        if verbose and it % 100 == 0:
            print ('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return loss_history

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: D x N array of training data. Each column is a D-dimensional point.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
        array of length N, and each element is an integer giving the predicted
        class.
        """
        y_pred = np.zeros(X.shape[1])
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################
        # pass
        scores = X.dot(self.W)
        y_pred = np.argmax(scores, axis = 1)
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred
    
    def load_weights(self, row, col, path = 'model\svm-scratch.sav'):
        self.W = np.loadtxt(path, encoding="utf-8").reshape(row, col)
    
    def save_weights(self, path = 'model\svm-scratch.sav'):
        weights = self.W
        print(weights.shape)
        with open(path, 'w') as save_file:
            for row in weights:
                np.savetxt(save_file, row)

    def loss(self, X_batch, y_batch, reg):
        """
        Compute the loss function and its derivative. 
        Subclasses will override this.

        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N
        data points; each point has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W
        """
        pass


class LinearSVM(LinearClassifier):
    """ A subclass that uses the Multiclass SVM loss function """

    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)
