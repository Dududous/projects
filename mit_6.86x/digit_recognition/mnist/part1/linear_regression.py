import numpy as np

### Functions for you to fill in ###

def closed_form(X, Y, lambda_factor):
    # Compute X transpose times X
    XT_X = np.dot(X.T, X)
    
    # Add lambda_factor times identity matrix to XT_X
    regularization_term = lambda_factor * np.identity(X.shape[1])
    XT_X_regularized = XT_X + regularization_term
    
    # Compute the inverse of XT_X_regularized
    XT_X_regularized_inv = np.linalg.inv(XT_X_regularized)
    
    # Compute X transpose times Y
    XT_Y = np.dot(X.T, Y)
    
    # Compute theta using the closed form solution
    theta = np.dot(XT_X_regularized_inv, XT_Y)
    
    return theta

### Functions which are already complete, for you to use ###

def compute_test_error_linear(test_x, Y, theta):
    test_y_predict = np.round(np.dot(test_x, theta))
    test_y_predict[test_y_predict < 0] = 0
    test_y_predict[test_y_predict > 9] = 9
    return 1 - np.mean(test_y_predict == Y)
