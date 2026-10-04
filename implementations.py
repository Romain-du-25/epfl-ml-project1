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

#
