import numpy as np


def inspect_data(X, name="Dataset"):
    """Print basic information about a feature matrix."""
    print(f"\n--- {name} ---")
    print("Shape:", X.shape)
    print("NaN values:", np.isnan(X).sum())
    print("Infinite values:", np.isinf(X).sum())

    missing_per_feature = np.isnan(X).sum(axis=0)
    constant_features = np.all(X == X[0], axis=0)

    print("Features containing NaN:",
          np.sum(missing_per_feature > 0))
    print("Constant features:", np.sum(constant_features))

    # Useful for identifying suspicious encoded values.
    for j in range(min(5, X.shape[1])):
        print(
            f"Feature {j}: "
            f"min={np.nanmin(X[:, j]):.3f}, "
            f"max={np.nanmax(X[:, j]):.3f}, "
            f"unique={len(np.unique(X[:, j]))}"
        )


def fit_preprocessor(X_train):
    """
    Learn imputation, feature selection and scaling parameters
    from the training data only.
    """
    X_train = np.asarray(X_train, dtype=float)

    if not np.all(np.isfinite(X_train) | np.isnan(X_train)):
        raise ValueError("Training data contains infinite values.")

    # 1. Remove features that are entirely missing.
    keep_nonempty = ~np.all(np.isnan(X_train), axis=0)
    X = X_train[:, keep_nonempty]

    # 2. Replace NaN with the training feature mean.
    means = np.nanmean(X, axis=0)
    X = np.where(np.isnan(X), means, X)

    # 3. Remove constant features.
    stds = np.std(X, axis=0)
    keep_variable = stds > 0

    # Save everything needed to transform other datasets.
    params = {
        "keep_nonempty": keep_nonempty,
        "keep_variable": keep_variable,
        "means": means,
        "stds": stds[keep_variable],
        "scaling_means": np.mean(X, axis=0)[keep_variable],
    }

    return params


def transform_data(X, params):
    """Apply the fitted preprocessing to another dataset."""
    X = np.asarray(X, dtype=float)

    # Use exactly the same selected features as training.
    X = X[:, params["keep_nonempty"]]

    # Use training means to impute missing values.
    X = np.where(np.isnan(X), params["means"], X)

    # Remove constant features identified during training.
    X = X[:, params["keep_variable"]]

    # Standardize using training statistics.
    X = (
        X - params["scaling_means"]
    ) / params["stds"]

    # Add the intercept column for logistic regression.
    X = np.column_stack((np.ones(X.shape[0]), X))

    return X