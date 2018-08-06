'''
Utils function that are helpful in constructing data science solutions.
Functions are supposed to be generic and kind of plug-and-play.
'''

import os
import time
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
#import zipfile
#import zlib


################# DATA PREPARATION #################################


def prepare_dtrains(X_train, Y_train, X_test, X_cols, missing=-9999):
    '''
    Returns DMatrices of given train & test data.
    '''
    dtrain = xgb.DMatrix(X_train, label=Y_train, missing=missing, feature_names=X_cols)
    dtest = xgb.DMatrix(X_test, missing=missing, feature_names=X_cols)
    print(dtrain.num_row(), dtrain.num_col())
    print(X_train.shape)
    print(dtest.num_row(), dtest.num_col())
    print(X_test.shape)
    return dtrain,dtest


def form_bins(df, colname, q=10):
    '''
    Creates binned column in the df corresponding to a given column.
    Binning is done on the basis of quantiles(given q).
    '''
    df['binned_'+colname], bins = pd.qcut(df[colname], q=q, labels=False, retbins=True)
    print('Bins: ', bins)
    return df


def create_combined_df(train, test):
    '''
    Creates combined dataframe from train & test dataframes.
    Basically concats train & test dataframes.
    
    Parameters
    -----------
    train : a pandas DataFrame.
    test : a pandas DataFrame.
    
    Returns
    --------
    combined_df : concated pandas dataframe.
    '''
    train['train_or_test'] = 'train'
    test['train_or_test'] = 'test'
    combined_df = pd.concat([train,test])
    combined_df.reset_index(drop=True, inplace=True)
    return combined_df


################## EDA ########################################################


def feat_stats(col):
    '''
    Expects that there is a combined_df dataframe formed from train & test dataframes and has 'train_or_test' column.
    '''
    print(combined_df[col].head(3), '\n')
    train_missing = combined_df[combined_df.train_or_test=='train'][col].isnull().sum()
    print('No of missing values in train: ', col, ' :', train_missing)
    print('Missing % : ', (train_missing*100/combined_df[combined_df.train_or_test=='train'].shape[0]))
    test_missing = combined_df[combined_df.train_or_test=='test'][col].isnull().sum()
    print('No of missing values in test: ', col, ' :', test_missing)
    print('Missing % : ', (test_missing*100/combined_df[combined_df.train_or_test=='test'].shape[0]), '\n')
    print('Other statistics:')
    print(combined_df[col].describe())


def findMissing(aSeries):
    """
    Gives total, non-missing, missing & percentage of missing observations in a pandas series.
    
    Parameters
    ----------
    aSeries : Pandas Series
    
    Returns
    -------
    missingList : List with 1st element denotes total number of observations in a series.
                  2nd element denotes total non-missing observations.
                  3rd element denotes total missing observations.
                  4th element denotes percentage of missing observations.
    """
    missingObs = aSeries.isnull().sum()
    nonMissingObs = aSeries.count()
    totalObs = len(aSeries)
    percentageMissing = round((float(missingObs) / float(totalObs) * 100),2)
    missingList = [totalObs,nonMissingObs,missingObs,percentageMissing]
    return missingList


def get_stats(group):
    return {'min': group.min(), 'max': group.max(), 'count': group.count(), 'mean': group.mean(), 'median':group.median(),
           'std': group.std(), '10_perc':group.quantile(q=0.1)}
# Use like
# gpby_obj.apply(get_stats).unstack()


def time_elapsed(t0, str_result=True):
    t = np.round((time.time()-t0)/60, 3)
    if str_result==True:
        return str(t)+' minutes elapsed!'
    else:
        return t
        
###################### FEATURE ENGINEERING #####################################


def generate_date_features(df, dateCol):
    '''
    Generates date based features from date column.
    
    Args:
    ----
    df: Pandas DataFrame.
    dateCol: Date Column Name in df.
    
    Returns:
    -------
    df: Pandas DataFrame with date features added.
    '''
    t0 = time.time()
    df['day'] = df[dateCol].dt.day
    df['month'] = df[dateCol].dt.month
    df['year'] = df[dateCol].dt.year
    df['dayofweek'] = df[dateCol].dt.dayofweek
    df['dayofyear'] = df[dateCol].dt.dayofyear
    df['week'] = df[dateCol].dt.week
    df['is_month_end'] = df[dateCol].dt.is_month_end
    df['is_month_start'] = df[dateCol].dt.is_month_start
    df['is_quarter_end'] = df[dateCol].dt.is_quarter_end
    df['is_quarter_start'] = df[dateCol].dt.is_quarter_start
    df['quarter'] = df['month'].apply(lambda mon: ceil(mon/3))
    print('Total time elapsed in making date features: ', (time.time()-t0)/60, 'minutes!')
    return df


def total_days_till_today(last_date):
    '''
    Returns total number of days between given last_date and today.
    '''
    return (pd.to_datetime('today') - last_date).days


def convert_dates(date):
    '''
    Converts date of type 01-Jan-17 into datetime object.
    '''
    return datetime.strptime(date, '%d-%b-%y')


def one_hot_encoder(df, ohe_cols):
    '''
    One-Hot Encoder function.
    
    Args:
    ----
    df: Pandas DataFrame 
    ohe_cols: Columns/Feature names that are to be one-hot encoded in df.
    
    Returns:
    -------
    df: Pandas DataFrame with one-hot encoded given features.
    '''
    print('Creating OHE features..\nOld df shape:{}'.format(df.shape))
    df = pd.get_dummies(df, columns=ohe_cols)
    print('New df shape:{}'.format(df.shape))
    return df


def create_lag_feats(df, gpby_cols, target_col, lags):
    '''
    Function to create lag features.
    
    Args:
    ----
    df: A Pandas DataFrame in which lags features to be created.
    gpby_cols: List of columns on which df should be grouped.
               Order of column names in list matters.
               Like ['store','item']
    target_col: String. Name of Target column whose lag values to be created.
                Like 'sales'
    lags: List. Lag values whose lag columns to be created.
          Like [7,14,28,91]
    
    Returns:
    -------
    df: Pandas DataFrame with lag features added.
    '''
    gpby = df.groupby(gpby_cols)
    for i in lags:
        df['_'.join([target_col, 'lag', str(i)])] = \
                gpby[target_col].shift(i).values + np.random.normal(scale=5.0, size=(len(df),))
    return df


def create_rmean_feats(df, gpby_cols, target_col, windows, min_periods=2, 
                             shift=1, win_type=None):
    '''
    Function to create rolling mean features.
    
    Args:
    ----
    df: A Pandas DataFrame in which rmean features to be created.
    gpby_cols: List of columns on which df should be grouped.
               Order of column names in list matters.
               Like ['store','item']
    target_col: String. Name of Target column on which rmean features would be based.
                Like 'sales'
    windows: List: Window numbers on which window would be rolled to create these features.
             Like [3,5,7]
    min_periods: Int. Minimum periods in which values should be present in a window
                 so as to calculate these rmean values.
    shift: Int. Target column to be shifted by this number so as to avoid current values.
    win_type: Categorical. Window Type (same as in pandas)
              Like 'triang'
         
    Returns:
    -------
    df: Pandas DataFrame with rmean features added.
    '''
    gpby = df.groupby(gpby_cols)
    for w in windows:
        df['_'.join([target_col, 'rmean', str(w)])] = \
            gpby[target_col].shift(shift).rolling(window=w, 
                                                  min_periods=min_periods,
                                                  win_type=win_type).mean().values +\
            np.random.normal(scale=5.0, size=(len(df),))
    return df


def create_sales_rmed_feats(df, gpby_cols, target_col, windows, min_periods=2, 
                            shift=1, win_type=None):
    '''
    Function to create rolling median features.
    
    Args:
    ----
    df: A Pandas DataFrame in which rmedian features to be created.
    gpby_cols: List of columns on which df should be grouped.
               Order of column names in list matters.
               Like ['store','item']
    target_col: String. Name of Target column on which rmedian features would be based.
                Like 'sales'
    windows: List: Window numbers on which window would be rolled to create these features.
             Like [3,5,7]
    min_periods: Int. Minimum periods in which values should be present in a window
                 so as to calculate these rmedian values.
    shift: Int. Target column to be shifted by this number so as to avoid current values.
    win_type: Categorical. Window Type (same as in pandas)
              Like 'triang'
         
    Returns:
    -------
    df: Pandas DataFrame with rmedian features added.
    '''
    gpby = df.groupby(gpby_cols)
    for w in windows:
        df['_'.join([target_col, 'rmed', str(w)])] = \
            gpby[target_col].shift(shift).rolling(window=w, 
                                                  min_periods=min_periods,
                                                  win_type=win_type).median().values +\
            np.random.normal(scale=5.0, size=(len(df),))
    return df


def create_sales_ewm_feats(df, gpby_cols, target_col, alpha=[0.9], shift=[1]):
    '''
    Function to create exponentially weighted mean features.
    
    Args:
    ----
    df: A Pandas DataFrame in which ewm features to be created.
    gpby_cols: List of columns on which df should be grouped.
               Order of column names in list matters.
               Like ['store','item']
    target_col: String. Name of Target column on which ewm features would be based.
                Like 'sales'
    alpha: list. List of alpha values on which ewm features to be calculated.
    shift: Int. Target column to be shifted by this number so as to avoid current values.
         
    Returns:
    -------
    df: Pandas DataFrame with ewm features added.
    '''
    gpby = df.groupby(gpby_cols)
    for a in alpha:
        for s in shift:
            df['_'.join([target_col, 'lag', str(s), 'ewm', str(a)])] = \
                gpby[target_col].shift(s).ewm(alpha=a).mean().values
    return df


#################### FEATURE SELECTION #########################################


def feats_importance_sklearn(colnames, mod):
    '''
    Gives Feature Importance combined with column names.
    model should have `feature_importances_` method as in sklearn's API.

    Args:
    -----
    colnames: ordered list of column names as passed to the model.
    mod: Model (rf or xgboost)

    Returns
    -------
    a sorted list(most important column first) of tuples of column names and their corresponding importance
    '''
    return sorted(zip(colnames, map(lambda x: round(x,5), mod.feature_importances_)), key=lambda x: x[1], reverse=True)


def feats_importance(model):
    '''
    Gives feature importance for xgboost and lightgbm models trained by their native APIs.
    
    Args:
    ----
    model: LightGBM/XGBoost model 
    
    Returns:
    -------
    feat_imp: A pandas DataFrame in descending order of gain received by model from each of the features.
    '''
    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature':model.feature_name(), 
                             'split':model.feature_importance('split'), 
                             'gain':100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    return feat_imp


#################### EVAL FUNCTIONS ############################################


def rmsle(preds, y_true):
    '''
    Root Mean Square Logarithmic Error
    '''
    y_true = y_true.get_label()
    n = preds.shape[0]
    return 'rmsle', np.sqrt(np.sum((np.log1p(preds)-np.log1p(y_true))**2)/n)


def smape(preds, target):
    '''
    Function to calculate SMAPE (Symmetric Mean Absolute Percent Error)
    '''
    n = len(preds)
    masked_arr = ~((preds==0)&(target==0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds-target)
    denom = np.abs(preds)+np.abs(target)
    smape_val = (200*np.sum(num/denom))/n
    return smape_val


def lgbm_smape(preds, train_data):
    '''
    Custom Evaluation Function to calculate SMAPE for LGBM
    '''
    labels = train_data.get_label()
    smape_val = smape(preds, labels)
    return 'SMAPE', smape_val, False


#################### MODELS #################################################

##### XGBOOST

# sample params

xgb_params ={'objective':'reg:linear', 'eta':0.01, 'max_depth':6, 'subsample':0.9, 'early_stopping_rounds':200, 'nrounds':5000,
             'colsample_bytree':0.7, 'booster':'gbtree', 'nthread':6, 'gamma':0.01, 'feval':rmsle, 'verbose_eval':100}

def xgb_validation(params, dtrain, dval, Y_val):
    t0 = time.time()
    model = xgb.train(params, dtrain, num_boost_round=params['nrounds'], evals=[(dtrain, 'train'), (dval, 'val')],
                      early_stopping_rounds=params['early_stopping_rounds'], verbose_eval=params['verbose_eval'])
    print(model.best_iteration)
    print('Total time taken to build the model: ', (time.time()-t0)/60, 'minutes!!')
    pred_Y_val = model.predict(dval)
    val_df = pd.DataFrame(columns=['true_Y_val','pred_Y_val'])
    val_df['pred_Y_val'] = pred_Y_val
    val_df['true_Y_val'] = Y_val
    print(val_df.shape)
    print(val_df.head())
    return model, val_df


def xgb_train(params, dtrain_all, dtest, num_round):
    t0 = time.time()
    model = xgb.train(params, dtrain_all, num_boost_round=num_round)
    test_preds = model.predict(dtest)
    print('Total time taken in model training: ', (time.time()-t0)/60, 'minutes!')
    return model, test_preds

##### LightGBM

# sample params

lgb_params ={'task':'train', 'boosting_type':'gbdt', 'objective':'binary', 'metric': {'auc', 'binary_logloss'},
             'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 0.9, 'bagging_fraction': 0.8, 'bagging_freq': 5,
             'verbose': 0, 'num_boost_round':5000, 'early_stopping_rounds':20, 'nthread':16}

def lgb_validation(params, lgbtrain, lgbval, X_val, Y_val, verbose_eval):
    t0 = time.time()
    evals_result = {}
    model = lgb.train(params, lgbtrain, num_boost_round=params['num_boost_round'], valid_sets=[lgbtrain, lgbval], 
                      early_stopping_rounds=params['early_stopping_rounds'], evals_result=evals_result, verbose_eval=verbose_eval)
    print(model.best_iteration)
    print('Total time taken to build the model: ', (time.time()-t0)/60, 'minutes!!')
    pred_Y_val = model.predict(X_val, num_iteration=model.best_iteration)
    val_df = pd.DataFrame(columns=['true_Y_val','pred_Y_val'])
    val_df['pred_Y_val'] = pred_Y_val
    val_df['true_Y_val'] = Y_val
    print(val_df.shape)
    print(val_df.head())
    return model, val_df


def lgb_train(params, lgbtrain_all, X_test, num_round):
    t0 = time.time()
    model = lgb.train(params, lgbtrain_all, num_boost_round=num_round)
    test_preds = model.predict(X_test, num_iteration=num_round)
    print('Total time taken in model training: ', (time.time()-t0)/60, 'minutes!')
    return model, test_preds



################### SUBMISSION PREPARATION ##########################################################################


def wavg(sub1, sub2, weights, id_col, outcome_col):
    '''
    
    '''
    merged_sub = sub1.merge(sub2, on=[id_col], how='left', suffixes=['_sub1','_sub2'])
    merged_sub[outcome_col] = (weights[0] * merged_sub[outcome_col+'_sub1'] + weights[1] * merged_sub[outcome_col+'_sub2'])
    return merged_sub[[id_col,outcome_col]]


def create_zip(path,fname,zipfname):
    '''
    Converts a file into zip file.
    
    Parameters
    ----------
    path: string. Path where file is located. This is the path of both file & zipped file.
    fname: string. Name of file.
    zipfname: string. Name that the zipped file should have.
    
    Returns
    -------
    None
    '''
    os.chdir(path)
    print('creating archive')
    zf = zipfile.ZipFile(zipfname,mode='w')
    try:
        zf.write(fname, compress_type=zipfile.ZIP_DEFLATED)
    finally:
        zf.close()
    print('Completed creating archive')


def df_to_zip(df, dirPath, fname):
    '''
    '''
    csvFileName = fname+'.csv'
    zipFileName = fname+'.zip'
    csvfilePath = dirPath+'/'+fname+'.csv'
    print("Creating csv file...")
    df.to_csv(csvfilePath, index=False)
    print("csv file created!")
    create_zip(dirPath, csvFileName, zipFileName)
    print("Removing csv file...")
    os.remove(csvfilePath)
    print("csv file removed! Only zip file remains.")
    

def extract_from_zip(inputZipFilePath,outputDir):
    '''
    Extracts a file from its zip file.
    
    Parameters
    ----------
    inputZipFilePath: string. Whole path with file name of the zipped file.
    outputDir: string. Path of the directory in which the zip file should be unzipped.
    
    Returns
    -------
    None
    '''
    zip_ref = zipfile.ZipFile(inputZipFilePath,'r')
    zip_ref.extractall(outputDir)
    zip_ref.close()
