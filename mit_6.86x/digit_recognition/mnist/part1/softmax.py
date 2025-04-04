import sys
sys.path.append("..")
import utils
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse


def augment_feature_vector(X):
    """
    Adds the x[i][0] = 1 feature for each data point x[i].

    Args:
        X - a NumPy matrix of n data points, each with d - 1 features

    Returns: X_augment, an (n, d) NumPy array with the added feature for each datapoint
    """
    column_of_ones = np.zeros([len(X), 1]) + 1
    return np.hstack((column_of_ones, X))

def compute_probabilities(X, theta, temp_parameter):
    """
    Computes, for each data point X[i], the probability that X[i] is labeled as j
    for j = 0, 1, ..., k-1.

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        theta - (k, d) NumPy array, where row j represents the parameters of our model for label j
        temp_parameter - the temperature parameter of the softmax function (scalar)

    Returns:
        H - (k, n) NumPy array, where each entry H[j][i] is the probability that X[i] is labeled as j
    """
    # Compute the dot product of theta and X.T, divided by temp_parameter
    logits = np.dot(theta, X.T) / temp_parameter
    
    # Subtract max(logits) for numerical stability
    logits_stable = logits - np.max(logits, axis=0)
    
    # Compute exponentials of stabilized logits
    exp_logits = np.exp(logits_stable)
    
    # Compute softmax probabilities
    probabilities = exp_logits / np.sum(exp_logits, axis=0)
    
    return probabilities


def compute_cost_function(X, Y, theta, lambda_factor, temp_parameter):
    """
    Computes the total cost over every datapoint.

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each data point
        theta - (k, d) NumPy array, where row j represents the parameters of our model for label j
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        c - the cost value (scalar)
    """
    # Number of data points
    n = X.shape[0]
    
    # Compute probabilities using the provided compute_probabilities function
    probabilities = compute_probabilities(X, theta, temp_parameter)
    
    # Create a mask for the correct class probabilities
    correct_class_probabilities = probabilities[Y, np.arange(n)]
    
    # Compute the log of correct class probabilities
    log_correct_class_probabilities = np.log(correct_class_probabilities)
    
    # Compute the first term of the cost function (cross-entropy loss)
    cross_entropy_loss = -np.sum(log_correct_class_probabilities) / n
    
    # Compute the second term of the cost function (L2 regularization)
    l2_regularization = (lambda_factor / 2) * np.sum(theta**2)
    
    # Total cost is the sum of cross-entropy loss and regularization term
    cost = cross_entropy_loss + l2_regularization
    
    return cost


def run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter):
    """
    Runs one step of batch gradient descent.

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each data point
        theta - (k, d) NumPy array, where row j represents the parameters of our model for label j
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        theta - (k, d) NumPy array that is updated after one step of gradient descent
    """
    # Number of classes (k) and number of data points (n)
    k = theta.shape[0]
    n = X.shape[0]
    
    # Compute probabilities using the compute_probabilities function
    probabilities = compute_probabilities(X, theta, temp_parameter)
    
    # Create a sparse matrix for the indicator function [Y == j]
    M = sparse.coo_matrix((np.ones(n), (Y, np.arange(n))), shape=(k, n)).toarray()
    
    # Compute the gradient of the cost function with respect to theta
    gradient = (-1 / (temp_parameter * n)) * np.dot(M - probabilities, X) + lambda_factor * theta
    
    # Update theta using gradient descent
    theta = theta - alpha * gradient
    
    return theta

def update_y(train_y, test_y):
    """
    Changes the old digit labels for the training and test set to the new (mod 3) labels.

    Args:
        train_y - (n, ) NumPy array containing the labels (a number between 0-9) 
                  for each datapoint in the training set
        test_y - (n, ) NumPy array containing the labels (a number between 0-9) 
                 for each datapoint in the test set

    Returns:
        train_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                       for each datapoint in the training set
        test_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                      for each datapoint in the test set
    """
    # Compute mod 3 for both train and test labels
    train_y_mod3 = train_y % 3
    test_y_mod3 = test_y % 3
    
    return train_y_mod3, test_y_mod3


def compute_test_error_mod3(X, Y, theta, temp_parameter):
    """
    Returns the error of these new labels when the classifier predicts the digit (mod 3).

    Args:
        X - (n, d - 1) NumPy array (n datapoints each with d - 1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-2) for each data point
        theta - (k, d) NumPy array, where row j represents the parameters of our model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        test_error - the error rate of the classifier (scalar)
    """
    # Get predicted classifications for X
    predicted_labels = get_classification(X, theta, temp_parameter)
    
    # Compute predicted labels mod 3
    predicted_labels_mod3 = predicted_labels % 3
    
    # Calculate error rate by comparing predicted mod 3 labels to true mod 3 labels
    test_error = 1 - np.mean(predicted_labels_mod3 == Y)
    
    return test_error

def softmax_regression(X, Y, temp_parameter, alpha, lambda_factor, k, num_iterations):
    """
    Runs batch gradient descent for a specified number of iterations on a dataset
    with theta initialized to the all-zeros array. Here, theta is a k by d NumPy array
    where row j represents the parameters of our model for label j for
    j = 0, 1, ..., k-1

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d-1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        temp_parameter - the temperature parameter of softmax function (scalar)
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        k - the number of labels (scalar)
        num_iterations - the number of iterations to run gradient descent (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
        cost_function_progression - a Python list containing the cost calculated at each step of gradient descent
    """
    X = augment_feature_vector(X)
    theta = np.zeros([k, X.shape[1]])
    cost_function_progression = []
    for i in range(num_iterations):
        cost_function_progression.append(compute_cost_function(X, Y, theta, lambda_factor, temp_parameter))
        theta = run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter)
    return theta, cost_function_progression

def get_classification(X, theta, temp_parameter):
    """
    Makes predictions by classifying a given dataset

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d - 1 features)
        theta - (k, d) NumPy array where row j represents the parameters of our model for
                label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        Y - (n, ) NumPy array, containing the predicted label (a number between 0-9) for
            each data point
    """
    X = augment_feature_vector(X)
    probabilities = compute_probabilities(X, theta, temp_parameter)
    return np.argmax(probabilities, axis = 0)

def plot_cost_function_over_time(cost_function_history):
    plt.plot(range(len(cost_function_history)), cost_function_history)
    plt.ylabel('Cost Function')
    plt.xlabel('Iteration number')
    plt.show()

def compute_test_error(X, Y, theta, temp_parameter):
    error_count = 0.
    assigned_labels = get_classification(X, theta, temp_parameter)
    return 1 - np.mean(assigned_labels == Y)
