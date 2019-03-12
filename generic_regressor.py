import pandas as pd
import numpy as np
from IPython.display import display

# Statistical packages used for transformations
from scipy import stats
import scipy.stats as ss
from scipy.stats import skew, norm
from scipy.special import boxcox1p
from scipy.stats.stats import pearsonr

# Algorithms used for modeling
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn import ensemble
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.kernel_ridge import KernelRidge
import xgboost as xgb

# Model selection packages used for sampling dataset and optimising parameters
from sklearn import model_selection
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit

# To ignore annoying warnings
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)
warnings.filterwarnings("ignore", category=DeprecationWarning)

print('All Modules Imported Succesfully!')
#print(sys.argv[1])

#DATA PREP

#load data
def load_data():
    train = pd.read_csv(sys.argv[1])
    test = pd.read_csv(sys.argv[2])

# Save the 'Id' column
    train_ID = train[sys.argv[3]]
    test_ID = test[sys.argv[3]]

# Now drop the  'Id' column as it's redundant for modeling
    train.drop(sys.argv[3], axis = 1, inplace = True)
    test.drop(sys.argv[3], axis = 1, inplace = True)
    return train,test,test_ID

train,test,test_ID = load_data()
print('Training data loaded: ', train.shape)
print('Test data loaded: ', test.shape)


#Remove outliers from target. Drop rows from the train data where the corresponding target value is an outlier

#reject outliers from target if the number falls above the 90th percentile
def reject_outliers(target):
    rowindx = list()
    prcntl = np.percentile(train[target], 90)
    for indx in range(0,len(train[target])):
        if train.loc[indx,target] > prcntl:
            rowindx.append(indx)
    train.drop(labels = rowindx, axis = 0,inplace = True)

#target from command line
target = sys.argv[4]
reject_outliers(target)
print('Outliers removed from target...')

#treat missing values in test and train data - Fill missing values in numeric cols with the mean and missing values in the non numeric cols with the mode

#First of all, save the length of the training and test data for use later
ntrain = train.shape[0]
ntest = test.shape[0]

# Also save the target value, as we will remove this
y_train = train[target]

# concatenate training and test data into all_data
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop([target], axis=1, inplace=True)

print("Combined data shape: {}".format(all_data.shape))

#Fill Numeric columns with the mean where missing values exist and fill Non-numeric cols with mode where missing values exist

#start with getting the names of all numeric columns
numeric_cols = all_data.select_dtypes(include = [np.number]).columns.tolist()

#fill na with mean of columns
for col in numeric_cols:
    all_data[col] = all_data[col].fillna(all_data[col].mean())

#get names of non numeric columns
non_numeric_cols = all_data.select_dtypes(exclude = [np.number]).columns.tolist()

#fill na with mode of columns
for col in non_numeric_cols:
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

print("Missing values in features filled...")

#Fill missing values in target with mean
y_train = y_train.fillna(y_train.mean())

print("Missing values in target filled...")

#FEATURE ENGINEERING

#Enumerate Categorical Columns. Categorical columns will be enumerated by giving categories numbers corresponding to their frequency rank.

#All maps list to map non-numeric cols
allmaps = list()

#enumerate columns
for col in non_numeric_cols:
    mydict = {}
    #construct dictionary to use for map
    keys,values = np.unique(all_data[col],return_counts = True)
    ranks = ss.rankdata(values)
    for x in range(0,len(ranks)):
        mydict[keys[x]] = ranks[x]
    allmaps.append(mydict)
    #use map
    all_data[col] = all_data[col].map(mydict)
print("Columns Enumerated...")

#Create Polynomials from highest correlated features (>0.4)

high_cor_cols = list()
#get high correlated features
for col in numeric_cols:
    if (train[col].corr(y_train) > 0.4):
        high_cor_cols.append(col)

print("Highest correlated features: ", high_cor_cols)

#make quadratics, cubes and squareroots
for col in numeric_cols:
    if (train[col].corr(y_train) > 0.4):
        high_cor_cols.append(col)
print("Polynomials made...")

#Treat skews in features. Transform features with skews less than -0.5 and greater than 0.5
skewed_feats = all_data.apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = skewed_feats[abs(skewed_feats) > 0.5]

skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lam)

print(skewness.shape[0],  "skewed numerical features have been Box-Cox transformed")

#Treat skew in target. If target has skew less than -0.5 or greater than 0.5 then we treat it
target_skew = y_train.skew()

print("Target skew before treating: " , target_skew)

if abs(y_train.skew()) > 0.5:
#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
    y_train = np.log1p(y_train)

print("Target skew after treating: " , y_train.skew())

#Drop columns with variance in the bottom 10%

#reject column if variance is in bottom 10%
index = range(0,(len(all_data.columns)))
columns = ['ColumnName','Var']
var_cols = pd.DataFrame(index=index,columns=columns)
var_cols['Var'] = list(all_data.var())
var_cols['ColumnName'] = all_data.columns
prcntl = np.percentile(var_cols['Var'], 10)

low_var_cols = list()

#construct a list for low variance columns only
for indx in range(0,(len(var_cols))):
    if var_cols.loc[indx,'Var'] < prcntl:
        low_var_cols.append(var_cols.loc[indx,'ColumnName'])

#now drop the low var columns
all_data = all_data.drop(labels = low_var_cols, axis= 1)

print("Columns with low variance dropped. All data shape: ",all_data.shape)

#MODELING

#Start by reducing number of features using XGBoost

# First, re-create the training and test datasets
train = all_data[:ntrain]
test = all_data[ntrain:]

print("Recreated training data after treating: ", train.shape)
print("Recreated test data after treating: ", test.shape)

#Use XGBoost

xgb_train = train.copy()
xgb_test = test.copy()

model = xgb.XGBRegressor()
model.fit(xgb_train, y_train)

# Allow the feature importances attribute to select the most important features
xgb_feat_red = SelectFromModel(model, prefit = True)

# Reduce estimation, validation and test datasets
xgb_train = xgb_feat_red.transform(xgb_train)
xgb_test = xgb_feat_red.transform(xgb_test)

print("Results of 'feature_importances_':")
print('X_train: ', xgb_train.shape, '\nX_test: ', xgb_test.shape)

# Next we want to sample our data before applying to the test data
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(xgb_train, y_train, test_size=0.3, random_state=42)

print('X_train: ', X_train.shape, '\nX_test: ', X_test.shape, '\nY_train: ', Y_train.shape, '\nY_test: ', Y_test.shape)


#TRAINING BASE MODELS
#Machine Learning Algorithm (MLA) Selection and Initialization
models = [KernelRidge(), ElasticNet(), Lasso(), GradientBoostingRegressor(), BayesianRidge(), LassoLarsIC(), RandomForestRegressor(), xgb.XGBRegressor()]

# First I will use ShuffleSplit as a way of randomising the cross validation samples.
shuff = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

#create table to compare MLA metrics
columns = ['Name', 'Parameters', 'Train Accuracy Mean', 'Test Accuracy']
before_model_compare = pd.DataFrame(columns = columns)

#index through models and save performance to table
row_index = 0
for alg in models:

    #set name and parameters
    model_name = alg.__class__.__name__
    before_model_compare.loc[row_index, 'Name'] = model_name
    before_model_compare.loc[row_index, 'Parameters'] = str(alg.get_params())
    
    alg.fit(X_train, Y_train)
    
    #score model with cross validation
    training_results = np.sqrt((-cross_val_score(alg, X_train, Y_train, cv = shuff, scoring= 'neg_mean_squared_error')).mean())
    test_results = np.sqrt(((Y_test-alg.predict(X_test))**2).mean())
    
    before_model_compare.loc[row_index, 'Train Accuracy Mean'] = (training_results)*100
    before_model_compare.loc[row_index, 'Test Accuracy'] = (test_results)*100
    
    row_index+=1
    print(row_index, alg.__class__.__name__, 'trained...')

decimals = 3
before_model_compare['Train Accuracy Mean'] = before_model_compare['Train Accuracy Mean'].apply(lambda x: round(x, decimals))
before_model_compare['Test Accuracy'] = before_model_compare['Test Accuracy'].apply(lambda x: round(x, decimals))
print("scores before optimisation:")
print(before_model_compare)

#Optimisation using grid search. To cut down computation time some hyperparameters were hard coded.
models = [KernelRidge(), ElasticNet(), Lasso(), GradientBoostingRegressor(), BayesianRidge(), LassoLarsIC(), RandomForestRegressor(), xgb.XGBRegressor()]

KR_param_grid = {'alpha': [0.1], 'coef0': [100], 'degree': [1], 'gamma': [None], 'kernel': ['polynomial']}
EN_param_grid = {'alpha': [0.001], 'copy_X': [True], 'l1_ratio': [0.6], 'fit_intercept': [True], 'normalize': [False], 
                         'precompute': [False], 'max_iter': [300], 'tol': [0.001], 'selection': ['random'], 'random_state': [None]}
LASS_param_grid = {'alpha': [0.0005], 'copy_X': [True], 'fit_intercept': [True], 'normalize': [False], 'precompute': [False], 
                    'max_iter': [300], 'tol': [0.01], 'selection': ['random'], 'random_state': [None]}
GB_param_grid = {'loss': ['huber'], 'learning_rate': [0.1], 'n_estimators': [300], 'max_depth': [3], 
                                        'min_samples_split': [0.0025], 'min_samples_leaf': [5]}
BR_param_grid = {'n_iter': [200], 'tol': [0.00001], 'alpha_1': [0.00000001], 'alpha_2': [0.000005], 'lambda_1': [0.000005], 
                 'lambda_2': [0.00000001], 'copy_X': [True]}
LL_param_grid = {'criterion': ['aic'], 'normalize': [True], 'max_iter': [100], 'copy_X': [True], 'precompute': ['auto'], 'eps': [0.000001]}
RFR_param_grid = {'n_estimators': [50], 'max_features': ['auto'], 'max_depth': [None], 'min_samples_split': [5], 'min_samples_leaf': [2]}
XGB_param_grid = {'max_depth': [3], 'learning_rate': [0.1], 'n_estimators': [300], 'booster': ['gbtree'], 'gamma': [0], 'reg_alpha': [0.1],
                  'reg_lambda': [0.7], 'max_delta_step': [0], 'min_child_weight': [1], 'colsample_bytree': [0.5], 'colsample_bylevel': [0.2],
                  'scale_pos_weight': [1]}
params_grid = [KR_param_grid, EN_param_grid, LASS_param_grid, GB_param_grid, BR_param_grid, LL_param_grid, RFR_param_grid, XGB_param_grid]

after_model_compare = pd.DataFrame(columns = columns)

row_index = 0
for alg in models:
    
    gs_alg = GridSearchCV(alg, param_grid = params_grid[0], cv = shuff, scoring = 'neg_mean_squared_error', n_jobs=-1)
    params_grid.pop(0)

    #set name and parameters
    model_name = alg.__class__.__name__
    after_model_compare.loc[row_index, 'Name'] = model_name
    
    gs_alg.fit(X_train, Y_train)
    gs_best = gs_alg.best_estimator_
    after_model_compare.loc[row_index, 'Parameters'] = str(gs_alg.best_params_)
    
    #score model with cross validation:
    after_training_results = np.sqrt(-gs_alg.best_score_)
    after_test_results = np.sqrt(((Y_test-gs_alg.predict(X_test))**2).mean())
    
    after_model_compare.loc[row_index, 'Train Accuracy Mean'] = (after_training_results)*100
    after_model_compare.loc[row_index, 'Test Accuracy'] = (after_test_results)*100
    
    row_index+=1
    print(row_index, alg.__class__.__name__, 'Optimised...')

decimals = 3
after_model_compare['Train Accuracy Mean'] = after_model_compare['Train Accuracy Mean'].apply(lambda x: round(x, decimals))
after_model_compare['Test Accuracy'] = after_model_compare['Test Accuracy'].apply(lambda x: round(x, decimals))
print("scores after optimisation:")
print(after_model_compare)


#Stacking -  make predictions and construct stack
models = [KernelRidge(), ElasticNet(), Lasso(), GradientBoostingRegressor(), BayesianRidge(), LassoLarsIC(), RandomForestRegressor(), xgb.XGBRegressor()]
names = ['KernelRidge', 'ElasticNet', 'Lasso', 'Gradient Boosting', 'Bayesian Ridge', 'Lasso Lars IC', 'Random Forest', 'XGBoost']
params_grid = [KR_param_grid, EN_param_grid, LASS_param_grid, GB_param_grid, BR_param_grid, LL_param_grid, RFR_param_grid, XGB_param_grid]
stacked_validation_train = pd.DataFrame()
stacked_test_train = pd.DataFrame()

row_index=0

for alg in models:
    
    gs_alg = GridSearchCV(alg, param_grid = params_grid[0], cv = shuff, scoring = 'neg_mean_squared_error', n_jobs=-1)
    params_grid.pop(0)
    
    gs_alg.fit(X_train, Y_train)
    gs_best = gs_alg.best_estimator_
    stacked_validation_train.insert(loc = row_index, column = names[0], value = gs_best.predict(X_test))
    print(row_index+1, alg.__class__.__name__, 'predictions added to stacking validation dataset...')
    
    stacked_test_train.insert(loc = row_index, column = names[0], value = gs_best.predict(xgb_test))
    print(row_index+1, alg.__class__.__name__, 'predictions added to stacking test dataset...')
    print("-"*50)
    names.pop(0)
    
    row_index+=1
    
print('STACKING COMPLETE')
#Round data to 2 decimal points
stacked_validation_train = stacked_validation_train.round(decimals=2)
stacked_validation_train['XGBoost'].apply(lambda x:round(x,2))
stacked_test_train = stacked_test_train.apply(lambda x:round(x,2))
stacked_test_train['XGBoost'] = stacked_test_train['XGBoost'].apply(lambda x:round(x,2))
print('Stacked data rounded to 2 decimal places...')

#TRAIN AND OPTIMISE BAGGING MODEL THEN MAKE PREDICTIONS
RF_param_grid={
            'max_depth': range(3,7),
            'n_estimators': (10, 50, 100, 1000),
        }

def rfr_model(X, y,stacked_test_train):

# Perform Grid-Search
    gsc = GridSearchCV(
        estimator=RandomForestRegressor(),
        param_grid={
            'max_depth': range(3,7),
            'n_estimators': (10, 50, 100, 1000),
        },
        cv=5, scoring='neg_mean_squared_error', verbose=0,n_jobs=-1)
    
    grid_result = gsc.fit(X, y)
    best_params = grid_result.best_params_
    
    rfr = RandomForestRegressor(max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"],random_state=False, verbose=False)
    
# Perform K-Fold CV
    scores = cross_val_score(rfr, X, y, cv=10, scoring='mean_squared_error')
    rfr.fit(X, y)
    pred = rfr.predict(stacked_test_train)

    return scores, pred

#train the bagging model and get scores
print("Training Bagging Model....")
bagging_scores,bagging_pred = rfr_model(stacked_validation_train,Y_test,stacked_test_train)
print("Bagging model RMSE: " , np.mean(np.sqrt(abs(bagging_scores))))#the scores are returned with signs flipped, refer: https://github.com/scikit-learn/scikit-learn/issues/2439


#TRAIN AND OPTIMISE BOOSTING MODEL THEN MAKE PREDICTIONS
def GB_model(X, y,stacked_test_train):
# Perform Grid-Search
    parameters = {
    "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    "max_depth":[3,5,8],
    "max_features":["log2","sqrt"],
    "subsample":[0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
     }
    gb = GradientBoostingRegressor()
    gsc = GridSearchCV(GB, parameters, cv=10, n_jobs=-1)
    
    grid_result = gsc.fit(X, y)
    pred = gsc.predict(stacked_test_train)
    best_params = grid_result.best_params_

# Perform K-Fold CV
    scores = cross_val_score(gsc, X, y, cv=10, scoring='mean_squared_error')

    return scores,pred
print("Training Boosting Model....Go grab something to eat, this might take awhile...")
GB = GradientBoostingRegressor()
GB_scores,GB_pred=GB_model(stacked_validation_train,Y_test,stacked_test_train)
print("Boosting model RMSE: " , np.mean(np.sqrt(list(abs(GB_scores)))))


#CONSTRUCT SUBMISSION FILES

#submission file 1, bagging only
bagging_sub = pd.DataFrame()
bagging_sub['Id'] = test_ID
bagging_sub['SalePrice'] = bagging_pred
bagging_sub.to_csv('bagging_submission.csv',index=False)

#submission file 2, boosting only
GB_sub = pd.DataFrame()
GB_sub['Id'] = test_ID
GB_sub['SalePrice'] = GB_pred
GB_sub.to_csv('boosting_submission.csv',index=False)

#submission file 3, boosting + bagging(50/50 split)
ensemble = bagging_pred*(0.5) + GB_pred*(0.5)
ensemble_df = pd.DataFrame()
ensemble_df['Id'] = test_ID
ensemble_df['SalePrice'] = ensemble
ensemble_df.to_csv('ensemble_submission.csv',index=False)

print("Submission files created!")

