# Necessary imports
from numpy import dot, zeros

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
        y_hat = dot(X[i], w) + b  # (n,)(n,) = scalar (see np.dot)
        cost += (y_hat - y[i]) ** 2  # scalar
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
    dj_dw = zeros((n,))
    dj_db = 0.

    for i in range(m):
        # Compute for each input data
        y_hat = dot(X[i], w) + b
        err = y_hat - y[i]
        for j in range(n):
            # Compute for each parameter w_i
            dj_dw[j] += err * X[i, j]
        # compute the b parameter
        dj_db += err
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_db, dj_dw


def gradient_descent(X,
                     y,
                     X_test,
                     y_test,
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
      X (ndarray (m,n))   : Data, m examples with n features (Train data)
      y (ndarray (m,))    : target values (Train data)
      X_test (ndarray (m,n))   : Data, m examples with n features (Test data)
      y_test (ndarray (m,))    : target values (Test data)
      w_in (ndarray (n,)) : initial model parameters
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent

    Returns:
      w (ndarray (n,)) : Updated values of parameters
      b (scalar)       : Updated value of parameter
      j_history        : Historic evolution of the error during the process
      test_history     : Historic error in the test set
      """

    # An array to store cost J and w's at each iteration primarily for graphing later
    j_history = []
    test_history = []
    w = deepcopy(w_in)  # avoid modifying global w within function
    b = b_in
    min_cost = 5000
    for i in range(num_iters+1):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w, b)

        # Update Parameters using previous w, b, alpha and the calculated gradient
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            j_history.append(cost_function(X, y, w, b))
            test_cost = cost_function(X_test, y_test, w, b)
            if test_cost < min_cost:
                min_cost = test_cost
                min_cost_iteration = i
            test_history.append(test_cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % ceil(num_iters / 10) == 0:
            print(f"Iteration {i:5d}: Cost {j_history[-1]:4.2f}. TEST {test_history[-1]:4.2f}")

    return w, b, j_history, test_history, min_cost, min_cost_iteration  # return final w,b and J history for graphing
