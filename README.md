
# EPFL ML Project 1 – Heart attack prediction

Machine Learning (CS-433), EPFL. Team: <name 1>, <name 2>, <name 3>.

The goal is to predict heart attack risk from health survey data (more than 300,000 individuals), using only NumPy implementations of the course methods. Predictions are scored on AIcrowd (F1 score).

## Repository structure

- `implementations.py`: the six required methods (`mean_squared_error_gd`, `mean_squared_error_sgd`, `least_squares`, `ridge_regression`, `logistic_regression`, `reg_logistic_regression`).
- `preprocessing.py`: data inspection, imputation of missing values, removal of constant features, standardization and intercept column. All statistics are fitted on the training data only.
- `helpers.py`: provided helpers to load the CSV files and write the submission file.
- `run.py`: full pipeline that reproduces our submission.

## Requirements

Python 3 with NumPy. No other library is needed to run the project (matplotlib is only allowed for visualization).

## How to reproduce our results

1. Download `x_train.csv`, `y_train.csv` and `x_test.csv` from the [AIcrowd challenge](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1) and put them, without renaming them, in `data/dataset/`.
2. From the repository root, run:

   ```
   python run.py
   ```

3. The script splits the training data (80% train, 20% validation, fixed seed), preprocesses it, trains logistic regression by gradient descent, tunes the decision threshold on the validation F1 score, and writes `submission.csv`.
4. Upload `submission.csv` on AIcrowd.

## Method summary

- Labels are converted from {-1, 1} to {0, 1} for training, and back to {-1, 1} for the submission.
- Missing values are replaced by the training mean, constant features are removed, and features are standardized.
- Logistic regression is trained with gradient descent. Because heart attacks are rare, the threshold is tuned on the validation set instead of using 0.5.
- Validation F1 is about 0.40, and the leaderboard F1 is 0.412 (accuracy 0.869) for the first submission.
