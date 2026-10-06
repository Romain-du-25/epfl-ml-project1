import numpy as np

# Hugo - mean_squarred_error_gd & mean_squarred_error_sgd


def _mse_loss(y, tx, w):
    e = y - tx @ w
    return 0.5 * np.mean(e**2)


def _mse_gradient(y, tx, w):
    e = y - tx @ w
    return -(tx.T @ e) / len(y)


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent."""
    w = initial_w.copy()

    for _ in range(max_iters):
        gradient = _mse_gradient(y, tx, w)
        w = w - gamma * gradient

    loss = _mse_loss(y, tx, w)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent."""
    w = initial_w.copy()

    for _ in range(max_iters):
        i = np.random.randint(len(y))

        y_i = y[i : i + 1]
        tx_i = tx[i : i + 1]

        gradient = _mse_gradient(y_i, tx_i, w)
        w = w - gamma * gradient

    loss = _mse_loss(y, tx, w)
    return w, loss


def least_squares(y, tx):
    """Compute the least-squares solution using the normal equations.

    Args:
        y: Target values as a one-dimensional array of shape (n_samples,).
        tx: Design matrix of shape (n_samples, n_features).

    Returns:
        A tuple ``(w, loss)`` containing the fitted weights and their MSE loss.
    """
    gram_matrix = tx.T @ tx
    right_hand_side = tx.T @ y
    w = np.linalg.solve(gram_matrix, right_hand_side)

    loss = _mse_loss(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Compute the ridge-regression solution using the normal equations.

    The optimized objective is the MSE plus ``lambda_ * ||w||^2``. As required
    by the project specification, the returned loss is only the MSE and does
    not include the regularization penalty.

    Args:
        y: Target values as a one-dimensional array of shape (n_samples,).
        tx: Design matrix of shape (n_samples, n_features).
        lambda_: Non-negative regularization strength.

    Returns:
        A tuple ``(w, loss)`` containing the fitted weights and their MSE loss.
    """
    n_samples, n_features = tx.shape
    regularizer = 2 * n_samples * lambda_ * np.eye(n_features)
    gram_matrix = tx.T @ tx + regularizer
    right_hand_side = tx.T @ y
    w = np.linalg.solve(gram_matrix, right_hand_side)

    loss = _mse_loss(y, tx, w)
    return w, loss
