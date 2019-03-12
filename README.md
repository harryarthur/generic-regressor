# Generic Regression Script for Kaggle

Completes data cleaning, feature engineering, model training, model optimisation, bagging, boosting, stacking and prediction generically. Good when a quick regression analysis is needed but not meant to replace comprehensive manual work. Although it produces submission files for kaggle, they're essentially just predictions (as they should be).

## Getting Started

The script needs 4 paramaters from the command line in the following order:

train data - path to train data. CSV only. The train data must have a unique identifier column (ID) and target column<br />
test data - path to test data. CSV only. The train data must have a unique identifier column (ID)<br />
ID - name of unique identifier column<br />
Target column - name of target column<br />

### Prerequisites

pandas, numpy, scipy, sklearn

## Running

To run from command line:

~$ python generic_regressor (path to train data) (path to test data) (name of ID column) (name of target column)

### Output
3 files produced on output:

A submission file with boosting only predictions a submission file with bagging only predictions and a submission file with from an ensemble of the two previous ones. 

### Task Structure
The script completes the following tasks in the following order:<br />

Loads test and train data into dataframes<br />
Removes rows where the target is above the 90th percentile<br />
Fill Numeric columns with the mean where missing values exist<br />
Fill Non-numeric cols with mode where missing values exist<br />
Enumerate all categorical columns<br />
Creates polynomials from highest correlated features<br />
Treat skews in features. Transform features with skews less than -0.5 and greater than 0.5<br />
Treat skew in target. If target has skew less than -0.5 or greater than 0.5 it is treated<br />
Drop columns with variance in bottom 10%<br />
Reduce features to important ones only<br />
Train 8 base models from scikitlearn and xgb (Kernel Ridge, Elastic Net, Lasso, Gradient Boosting regressor, bayesian ridge, lassolarsIC, random forrest regressor and xgb.XGBRegressor)<br />
Optimise the 8 base models<br />
Predict using the 8 base models<br />
Stack the predictions<br />
Normalize predictions<br />
Train, optimise bagging model on stack<br />
Train, optimise boosting model on stack<br />
Make final predictions using boosting model<br />
Make final predicitons using bagging model<br />
Make final predictions using an ensemble of the two<br />
Output 3 files<br />

## Authors

* **Harry Robinson** 


