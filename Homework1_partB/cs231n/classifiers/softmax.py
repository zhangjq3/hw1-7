from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train=X.shape[0]
    num_classes=W.shape[1]
    for i in range(num_train):
        scores=X[i,:].dot(W)-np.max(X[i,:].dot(W))
        y_label=y[i]
        prob=np.exp(scores[y_label])/np.sum(np.exp(scores))
        loss+=-1*np.log(prob)
        for j in xrange(num_classes):
            if j!=y[i]:
                dW[:,j]+=np.exp(scores[j]/np.sum(np.exp(scores)))*X[i]
    loss/=num_train
    loss += 0.5*reg*np.sum(np.square(W))
    dW = dW/num_train + reg*W
       

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train=X.shape[0]
    scores = np.dot(X,W)-np.max(np.dot(X,W),axis=1,keepdims=True)
    p=np.exp(scores)/np.sum(np.exp(scores),axis=1,keepdims=True)
    y_trueClass=np.zeros_like(p)
    y_trueClass[range(num_train),y]=1.0
    loss= -1*np.sum(y_trueClass*np.log(p))/num_train+0.5*reg*np.sum(W*W)
    dW = np.dot(X.T,p - y_trueClass) / num_train + reg * W

    
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
