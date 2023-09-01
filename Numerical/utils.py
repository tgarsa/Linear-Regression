# Necessary imports
import numpy as np  # np.dot

# Additional tools
from copy import deepcopy
from math import ceil


def compute_cost(X, y, w, b):
    """
    Compute cost function.
    Quadratic error.
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters
      b (scalar)       : model parameter

    Returns:
      cost (scalar): cost
    """

    # number of training examples
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        # Calculate the error/cost or each data. And sum the whole of them
        y_hat = np.dot(X[i], w) + b  # (n,)(n,) = scalar (see np.dot)
        cost = cost + (y_hat - y[i]) ** 2  # scalar
    cost = cost / (2 * m)  # scalar
    return cost


def compute_gradient(X, y, w, b):
    """
    Computes the gradient for linear regression, our model.
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters
      b (scalar)       : model parameter

    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w.
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b.
    """
    m, n = X.shape  # (number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):
        # Compute for each input data
        y_hat = np.dot(X[i], w) + b
        err = y_hat - y[i]
        for j in range(n):
            # Compute for each parameter w_i
            dj_dw[j] = dj_dw[j] + err * X[i, j]
        # compute for the b parameter
        dj_db = dj_db + err
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_db, dj_dw


def gradient_descent(X,
                     y,
                     w_in,
                     b_in,
                     cost_function,
                     gradient_function,
                     alpha,
                     num_iters):
    """
    Performs batch gradient descent to learn w and b. Updates w and b by taking
    num_iters gradient steps with learning rate alpha

    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent

    Returns:
      w (ndarray (n,)) : Updated values of parameters
      b (scalar)       : Updated value of parameter
      """

    # An array to store cost J and w's at each iteration primarily for graphing later
    j_history = []
    w = deepcopy(w_in)  # avoid modifying global w within function
    b = b_in
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w, b)

        # Update Parameters using previous w, b, alpha and the calculated gradient
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            j_history.append(cost_function(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {j_history[-1]:8.2f}   ")

    return w, b, j_history  # return final w,b and J history for graphing
