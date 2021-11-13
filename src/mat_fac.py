import random
import torch
from d2l import torch as d2l

import numpy as np
import pandas as pd

import os

# replace this with the recsys model
def alfa_only(X, alfa):  #@save
    """The linear regression model."""
    return alfa.repeat(len(X)).reshape(X[:,0].shape)

# replace this with the recsys model
def alfa_user(X, alfa, beta_u):  #@save
    """The linear regression model."""
    return alfa + beta_u[X[:,0]]

# replace this with the recsys model
def alfa_item(X, alfa, beta_i):  #@save
    """The linear regression model."""
    return alfa + beta_i[X[:,1]]

# replace this with the recsys model
def whole_nine_yards(X, alfa, beta_u, beta_i):  #@save
    """The linear regression model."""
    return alfa + beta_u[X[:,0]] + beta_i[X[:,1]]

def squared_loss(y_hat, y):  #@save
    """Squared loss."""
    return (y_hat - y.reshape(y_hat.shape))**2 / 2

def sgd(params, lr, batch_size):  #@save
    """Minibatch stochastic gradient descent."""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()