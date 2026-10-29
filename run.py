import numpy as np
from pathlib import Path

from helpers import load_csv_data, create_csv_submission
from preprocessing import (
    inspect_data,
    fit_preprocessor,
    transform_data,
)
from implementations import logistic_regression


def f1_score(y_true, y_pred):
    """F1 score for labels in {0, 1}."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def predict_proba(tx, w):
    """Probability of class 1 for each row of tx."""
    return 1.0 / (1.0 + np.exp(-np.clip(tx @ w, -500, 500)))


# ============================================================
# 1. Load the data
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "dataset"

print("Looking for CSV files in:", DATA_DIR)

if not DATA_DIR.exists():
    raise FileNotFoundError(f"Dataset folder not found: {DATA_DIR}")

print("Files found:", [f.name for f in DATA_DIR.iterdir()])

X_train, X_test, y_train, train_ids, test_ids = load_csv_data(str(DATA_DIR))

print("\nOriginal data shapes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)


# ============================================================
# 2. Inspect the original data
# ============================================================

inspect_data(X_train, "Original training data")
inspect_data(X_test, "Original test data")


# ============================================================
# 3. Split training data into training and validation sets
# ============================================================

rng = np.random.default_rng(42)
indices = rng.permutation(len(y_train))

split = int(0.8 * len(indices))

train_idx = indices[:split]
val_idx = indices[split:]

X_tr_raw = X_train[train_idx]
X_val_raw = X_train[val_idx]

# Convert labels from {-1, 1} to {0, 1}
y_tr = (y_train[train_idx] == 1).astype(int)
y_val = (y_train[val_idx] == 1).astype(int)

print("\nData split:")
print("Training subset:", X_tr_raw.shape)
print("Validation subset:", X_val_raw.shape)
print("Training labels:", y_tr.shape)
print("Validation labels:", y_val.shape)


# ============================================================
# 4. Fit preprocessing on training data only
# ============================================================

params = fit_preprocessor(X_tr_raw)

# Apply the same preprocessing parameters to every dataset.
X_tr = transform_data(X_tr_raw, params)
X_val = transform_data(X_val_raw, params)
X_test_processed = transform_data(X_test, params)

print("\nProcessed data shapes:")
print("Training:", X_tr.shape)
print("Validation:", X_val.shape)
print("Test:", X_test_processed.shape)


# ============================================================
# 5. Train logistic regression
# ============================================================

initial_w = np.zeros(X_tr.shape[1])

w, train_loss = logistic_regression(
    y_tr,
    X_tr,
    initial_w,
    max_iters=1000,
    gamma=0.01,
)

print("\nModel training:")
print("Training loss:", train_loss)


# ============================================================
# 6. Evaluate on the validation set and tune the threshold
# ============================================================

val_probabilities = predict_proba(X_val, w)

best_threshold, best_f1 = 0.5, 0.0
for threshold in np.linspace(0.01, 0.9, 90):
    f1 = f1_score(y_val, (val_probabilities >= threshold).astype(int))
    if f1 > best_f1:
        best_threshold, best_f1 = threshold, f1

y_val_pred = (val_probabilities >= best_threshold).astype(int)

print("\nValidation results:")
print("Best threshold:", best_threshold)
print("Validation F1:", best_f1)
print("Validation accuracy:", np.mean(y_val_pred == y_val))

true_positives = np.sum((y_val == 1) & (y_val_pred == 1))
true_negatives = np.sum((y_val == 0) & (y_val_pred == 0))
false_positives = np.sum((y_val == 0) & (y_val_pred == 1))
false_negatives = np.sum((y_val == 1) & (y_val_pred == 0))

print("\nConfusion matrix counts:")
print("True positives:", true_positives)
print("True negatives:", true_negatives)
print("False positives:", false_positives)
print("False negatives:", false_negatives)


# ============================================================
# 7. Generate the submission file
# ============================================================

test_probabilities = predict_proba(X_test_processed, w)
y_test_pred = np.where(test_probabilities >= best_threshold, 1, -1)

print("\nCompetition test predictions:")
print("Unique predictions:", np.unique(y_test_pred))
print("Number of positive predictions:", np.sum(y_test_pred == 1))

create_csv_submission(test_ids, y_test_pred, "submission.csv")
print("Saved submission.csv")

""" # ============================================================
# 6. Evaluate on the validation set
# ============================================================

logits = X_val @ w

# Numerically stable enough for prediction, even for large logits.
probabilities = 1.0 / (
    1.0 + np.exp(-np.clip(logits, -500, 500))
)

y_val_pred = (probabilities >= 0.5).astype(int)

accuracy = np.mean(y_val_pred == y_val)

print("\nValidation results:")
print("Validation accuracy:", accuracy)

# Additional evaluation: confusion matrix
true_positives = np.sum((y_val == 1) & (y_val_pred == 1))
true_negatives = np.sum((y_val == 0) & (y_val_pred == 0))
false_positives = np.sum((y_val == 0) & (y_val_pred == 1))
false_negatives = np.sum((y_val == 1) & (y_val_pred == 0))

print("\nConfusion matrix counts:")
print("True positives:", true_positives)
print("True negatives:", true_negatives)
print("False positives:", false_positives)
print("False negatives:", false_negatives)


# ============================================================
# 7. Check the competition test predictions (optional)
# ============================================================

test_logits = X_test_processed @ w

test_probabilities = 1.0 / (
    1.0 + np.exp(-np.clip(test_logits, -500, 500))
)

# Convert probabilities to {-1, 1}, as required for submission.
y_test_pred = np.where(test_probabilities >= 0.5, 1, -1)

print("\nCompetition test predictions:")
print("Prediction shape:", y_test_pred.shape)
print("Unique predictions:", np.unique(y_test_pred))
print("Number of positive predictions:", np.sum(y_test_pred == 1))
print("Number of negative predictions:", np.sum(y_test_pred == -1)) """
