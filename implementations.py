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
    """Compute the least-squares solution using the normal equations."""
    
    gram_matrix = tx.T @ tx
    right_hand_side = tx.T @ y
    w = np.linalg.solve(gram_matrix, right_hand_side)

    loss = _mse_loss(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Compute the ridge-regression solution using the normal equations."""
    
    n_samples, n_features = tx.shape
    regularizer = 2 * n_samples * lambda_ * np.eye(n_features)
    gram_matrix = tx.T @ tx + regularizer
    right_hand_side = tx.T @ y
    w = np.linalg.solve(gram_matrix, right_hand_side)

    loss = _mse_loss(y, tx, w)
    return w, loss

def _sigmoid(t):
    out = np.empty_like(t, dtype=float) #fill with garbage to avoid unnecessary allocations
    pos = t >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-t[pos]))
    exp_t = np.exp(t[~pos])
    out[~pos] = exp_t / (1.0 + exp_t)
    return out


def _logistic_loss(y, tx, w):
    """Mean negative log-likelihood."""
    z = tx @ w
    return np.mean(np.logaddexp(0, z) - y * z) #computes log(1 + exp(z)) in a numerically stable way


def _logistic_gradient(y, tx, w):
    """Gradient of the mean negative log-likelihood."""
    return tx.T @ (_sigmoid(tx @ w) - y) / len(y)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent (y in {0,1})."""
    w = initial_w.astype(float)

    for _ in range(max_iters):
        gradient = _logistic_gradient(y, tx, w)
        w = w - gamma * gradient

    loss = _logistic_loss(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent (y in {0,1}).

    Penalty is lambda_ * ||w||^2, so its gradient is 2 * lambda_ * w.
    The returned loss does NOT include the penalty term.
    """
    w = initial_w.astype(float)

    for _ in range(max_iters):
        gradient = _logistic_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * gradient

    loss = _logistic_loss(y, tx, w)
    return w, loss