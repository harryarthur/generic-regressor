# Generic Regression Script for Kaggle

Completes data cleaning, feature engineering, model training, model optimisation, bagging, boosting, stacking and prediction generically. Good when a quick regression analysis is needed but not meant to replace comprehensive manual work. Although it produces submission files for kaggle, they're essentially just predictions (as they should be).

## Getting Started

The script needs 4 paramaters from the command line in the following order:

train data - path to train data. CSV only. The train data must have a unique indentifier column (ID) and target column
test data - path to test data. CSV only. The train data must have a unique indentifier column (ID)
ID - name of unique identifier column
Target column - name of target column

### Prerequisites

pandas, numpy, scipy, sklearn

## Running

To run from command line:

~$ python generic_regressor (path to train data) (path to test data) (name of ID column) (name of target column)

### Output
3 files produced on output:

A submission file with boosting only predictions a submission file with bagging only predictions and a submission file with from an ensemble of the two previous ones. 

### Task Structure
The script completes the following tasks in the following order:

Loads test and train data into dataframes
Removes rows where the target is above the 90th percentile
Fill Numeric columns with the mean where missing values exist
Fill Non-numeric cols with mode where missing values exist
Enumerate all categorical columns
Creates polynomials from highest correlated features
Treat skews in features. Transform features with skews less than -0.5 and greater than 0.5
Treat skew in target. If target has skew less than -0.5 or greater than 0.5 it is treated
Drop columns with variance in bottom 10%
Reduce features to important ones only
Train 8 base models from scikitlearn and xgb (Kernel Ridge, Elastic Net, Lasso, Gradient Boosting regressor, bayesian ridge, lassolarsIC, random forrest regressor and xgb.XGBRegressor)
Optimise the 8 base models
Predict using the 8 base models
Stack the predictions
Normalize predictions
Train, optimise bagging model on stack
Train, optimise boosting model on stack
Make final predictions using boosting model
Make final predicitons using bagging model
Make final predictions using an ensemble of the two
Output 3 files

## Authors

* **Harry Robinson** 


