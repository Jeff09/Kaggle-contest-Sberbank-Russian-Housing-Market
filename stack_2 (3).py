# -*- coding: utf-8 -*-

from feature_enginerring import clean_data, add_features, add_spearman_features,add_decomposition_features, handle_missing_value_subarea, handle_missing_value_mostfreq
import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing
import xgboost as xgb
#import lightgbm as lgb

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVR
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.neural_network import MLPRegressor
from scipy.stats import norm
import copy

# Parameters
prediction_stderr = 0.0073  #  assumed standard error of predictions
                          #  (smaller values make output closer to input)
train_test_logmean_diff = 0.1  # assumed shift used to adjust frequencies for time trend
probthresh = 90  # minimum probability*frequency to use new price instead of just rounding
rounder = 2  # number of places left of decimal point to zero

macro_humility_factor = 0.7


class StackingCVRegressorRetrained(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, regressors, meta_regressor, n_folds=2, use_features_in_secondary=False):
        self.regressors = regressors
        self.meta_regressor = meta_regressor
        self.n_folds = n_folds
        self.use_features_in_secondary = use_features_in_secondary
        self.regr_ = [clone(x) for x in self.regressors]
        self.meta_regr_ = clone(self.meta_regressor)

    def fit(self, train, y):
        if type(train) == pd.core.frame.DataFrame:            
            X = copy.deepcopy(train).values
            #ADD more features for meta regressor
            #train = add_features(train)
        elif type(train) == np.ndarray:            
            X = copy.deepcopy(train)

        kfold = KFold(n_splits=self.n_folds, shuffle=True)

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.regressors)))

        # Create out-of-fold predictions for training meta-model
        for i, regr in enumerate(self.regr_):
            for train_idx, holdout_idx in kfold.split(X, y):
                instance = clone(regr)
                instance.fit(X[train_idx], y[train_idx])
                out_of_fold_predictions[holdout_idx, i] = instance.predict(X[holdout_idx])
       
        # Retrain base models on all data
        all_predictions = np.zeros((X.shape[0], len(self.regressors)))
        for i, regr in enumerate(self.regr_):
            regr.fit(X, y)
            all_predictions[:, i] = regr.predict(X)
        
        # Train meta-model
        #ADD more features
        if type(train) == pd.core.frame.DataFrame:
            train = add_features(train)
            X = train.values
        #X = train.values
        
        if self.use_features_in_secondary:
            self.meta_regr_.fit(np.hstack((X, out_of_fold_predictions)), y)
        else:
            self.meta_regr_.fit(out_of_fold_predictions, y)

        return self
    
    def predict(self, test):
        if type(test) == pd.core.frame.DataFrame:
            X = copy.deepcopy(test).values
        elif type(test) == np.ndarray:            
            X = copy.deepcopy(test)
        
        meta_features = np.column_stack([
            regr.predict(X) for regr in self.regr_
        ])
        
        if type(test) == pd.core.frame.DataFrame:
            #ADD more features
            test = add_features(test)        
            X = test.values

        if self.use_features_in_secondary:
            return self.meta_regr_.predict(np.hstack((X, meta_features)))
        else:
            return self.meta_regr_.predict(meta_features)
  
def stack():
    train = pd.read_csv('data/train.csv', parse_dates=['timestamp'])
    test = pd.read_csv('data/test.csv', parse_dates=['timestamp'])
    
    id_test = test.id
        
    train, test = clean_data(train, test)
    #train, test = add_features(train, test)
    #train, test = handle_missing_value_mostfreq(train, test)
    #train, test = add_spearman_features(train, test)
    train, test = add_decomposition_features(train, test)
    
    mult = .969
    #y_train = train["price_doc"] * mult + 10
    
    y_train = train['price_doc'].values * mult + 10
    y_mean = np.mean(y_train)
    
    train = train.drop(["id", "timestamp", "price_doc"], axis=1)#"average_q_price"
    #x_test = test.drop(["id", "timestamp", "average_q_price"], axis=1)
    test = test.drop(["id", "timestamp"], axis=1)
    
    num_train = len(train)
    x_all = pd.concat([train, test])
    
    for c in x_all.columns:
        if x_all[c].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(x_all[c].values))
            x_all[c] = lbl.transform(list(x_all[c].values))
    
    train = x_all[:num_train]
    test = x_all[num_train:]    
    
    print(train.shape, test.shape)
    
    en = make_pipeline(RobustScaler(), SelectFromModel(Lasso(alpha=0.03)), ElasticNet(alpha=0.001, l1_ratio=0.1))
    
    adb = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=300, random_state=2017)
    
    bag = BaggingRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=300, random_state=2017)
        
    rf = RandomForestRegressor(n_estimators=250, n_jobs=4, min_samples_split=25, min_samples_leaf=25)
                               
    et = ExtraTreesRegressor(n_estimators=250, n_jobs=4, min_samples_split=25, min_samples_leaf=25)
    
    gbr = GradientBoostingRegressor(loss='huber', learning_rate=0.1, n_estimators=350, min_samples_split=25, min_samples_leaf=25)
    
    xgbm = xgb.sklearn.XGBRegressor(max_depth=6, learning_rate=0.005, subsample=0.6, base_score=y_mean,
                                    objective='reg:linear', n_estimators=1000)
    
    nn = MLPRegressor(hidden_layer_sizes=(200, 400, 50), random_state =2017, early_stopping=True)
    
    svm = SVR(kernel='poly', degree=3, gamma='auto', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
    '''
    results = cross_val_score(gbr, train.values, y_train, cv=5, scoring='r2')#0.673187 (0.066807)
    print("gradient boosting regressor score: {:4f} ({:4f})".format(results.mean(), results.std()))
    
    results = cross_val_score(adb, train.values, y_train, cv=5, scoring='r2')#-0.373219 (0.343094)
    print("adboost regressor score: {:4f} ({:4f})".format(results.mean(), results.std()))
    
    results = cross_val_score(bag, train.values, y_train, cv=5, scoring='r2')#0.567878 (0.058145)
    print("bagging regressor score: {:4f} ({:4f})".format(results.mean(), results.std()))
    
    results = cross_val_score(nn, train.values, y_train, cv=5, scoring='r2')#-17.162579 (34.638616)
    print("neural network score: {:4f} ({:4f})".format(results.mean(), results.std()))
    
    results = cross_val_score(rf, train.values, y_train, cv=5, scoring='r2')#0.518(0.057)
    print("RandomForest score: {:4f} ({:4f})".format(results.mean(), results.std()))
    
    results = cross_val_score(et, train.values, y_train, cv=5, scoring='r2')#0.617(0.05)
    print("ExtraTrees score: {:4f} ({:4f})".format(results.mean(), results.std()))
    
    results = cross_val_score(xgbm, train.values, y_train, cv=5, scoring='r2')
    print("XGBoost score: {:4f} ({:4f})".format(results.mean(), results.std()))#0.658(0.065)
    
    
    #results = cross_val_score(svm, train.values, y_train, cv=5, scoring='r2')#
    #print("SVM score: {:4f} ({:4f})".format(results.mean(), results.std()))
            
    
    stack_with_feats = StackingCVRegressorRetrained((nn, rf, et), xgbm, use_features_in_secondary=True)
    
    results = cross_val_score(stack_with_feats, train.values, y_train, cv=5, scoring='r2')#en: 0.674925(0.06)
    print("Stacking (with primary feats) score: {:4f} ({:4f})".format(results.mean(), results.std()))
    #Stacking (with primary feats) nn score: 0.674353 (0.059497)
    '''    
    #stack_with_feats_2 = StackingCVRegressorRetrained((bag, gbr, rf, et), xgbm, use_features_in_secondary=True)   
    #results = cross_val_score(stack_with_feats_2, train.values, y_train, cv=5, scoring='r2')#0.675728 (0.066580)
    #print("Stacking (with primary feats) 2 score: {:4f} ({:4f})".format(results.mean(), results.std()))    
    
    stack_with_feats_2 = StackingCVRegressorRetrained((gbr, et), xgbm, use_features_in_secondary=True)  
    
    results = cross_val_score(stack_with_feats_2, train.values, y_train, cv=5, scoring='r2')#0.675728 (0.066580)
    print("Stacking (with primary feats) 2 score: {:4f} ({:4f})".format(results.mean(), results.std())) 
    
    #stack_with_feats_2 = StackingCVRegressorRetrained([xgbm, gbr], xgbm, use_features_in_secondary=False)   
    #results = cross_val_score(stack_with_feats_2, train.values, y_train, cv=5, scoring='r2')#0.675728 (0.066580)
    #print("Stacking (with primary feats) 2 score: {:4f} ({:4f})".format(results.mean(), results.std()))
    
    #stack_with_feats_2 = StackingCVRegressorRetrained((bag, xgbm, rf, et), gbr, use_features_in_secondary=True)
    
    #stack_with_feats_2 = StackingCVRegressorRetrained([xgbm], xgbm, use_features_in_secondary=True)#, gbr
    #results = cross_val_score(stack_with_feats_2, train.values, y_train, cv=5, scoring='r2')#0.675728 (0.066580)
    #print("Stacking (with primary feats) 2 score: {:4f} ({:4f})".format(results.mean(), results.std()))

    stack_with_feats_2.fit(train, y_train)
    y_pred = stack_with_feats_2.predict(test)
    df_sub = pd.DataFrame({'id': id_test, 'price_doc': y_pred})
    
    df_sub.to_csv('stack_2.csv', index=False)
    return df_sub

def model2():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    id_test = test.id
    
    mult = .969
    
    y_train = train["price_doc"] * mult + 10
    x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)
    x_test = test.drop(["id", "timestamp"], axis=1)
    
    for c in x_train.columns:
        if x_train[c].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(x_train[c].values))
            x_train[c] = lbl.transform(list(x_train[c].values))
    
    for c in x_test.columns:
        if x_test[c].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(x_test[c].values))
            x_test[c] = lbl.transform(list(x_test[c].values))
    
    xgb_params = {
        'eta': 0.05,
        'max_depth': 5,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1
    }
    
    dtrain = xgb.DMatrix(x_train, y_train)
    dtest = xgb.DMatrix(x_test)
    
    num_boost_rounds = 385  # This was the CV output, as earlier version shows
    model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)
    
    y_predict = model.predict(dtest)
    output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})
    
    
    df_train = pd.read_csv("data/train.csv", parse_dates=['timestamp'])
    df_test = pd.read_csv("data/test.csv", parse_dates=['timestamp'])
    df_macro = pd.read_csv("data/macro.csv", parse_dates=['timestamp'])
    
    df_train.drop(df_train[df_train["life_sq"] > 7000].index, inplace=True)
    
    mult = 0.969
    y_train = df_train['price_doc'].values * mult + 10
    id_test = df_test['id']
    
    df_train.drop(['id', 'price_doc'], axis=1, inplace=True)
    df_test.drop(['id'], axis=1, inplace=True)
    
    num_train = len(df_train)
    df_all = pd.concat([df_train, df_test])
    # Next line just adds a lot of NA columns (becuase "join" only works on indexes)
    # but somewhow it seems to affect the result
    df_all = df_all.join(df_macro, on='timestamp', rsuffix='_macro')
    print(df_all.shape)
    
    # Add month-year
    month_year = (df_all.timestamp.dt.month*30 + df_all.timestamp.dt.year * 365)
    month_year_cnt_map = month_year.value_counts().to_dict()
    df_all['month_year_cnt'] = month_year.map(month_year_cnt_map)
    
    # Add week-year count
    week_year = (df_all.timestamp.dt.weekofyear*7 + df_all.timestamp.dt.year * 365)
    week_year_cnt_map = week_year.value_counts().to_dict()
    df_all['week_year_cnt'] = week_year.map(week_year_cnt_map)
    
    # Add month and day-of-week
    df_all['month'] = df_all.timestamp.dt.month
    df_all['dow'] = df_all.timestamp.dt.dayofweek
    
    # Other feature engineering
    df_all['rel_floor'] = df_all['floor'] / df_all['max_floor'].astype(float)
    df_all['rel_kitch_sq'] = df_all['kitch_sq'] / df_all['full_sq'].astype(float)
    
    
    ######## BEGIN 2ND SET OF BILL S CHANGES 
    
    ## same ones as above 
    df_all['area_per_room'] = df_all['life_sq'] / df_all['num_room'].astype(float)
    df_all['livArea_ratio'] = df_all['life_sq'] / df_all['full_sq'].astype(float)
    df_all['yrs_old'] = 2017 - df_all['build_year'].astype(float)
    df_all['avgfloor_sq'] = df_all['life_sq']/df_all['max_floor'].astype(float) #living area per floor
    df_all['pts_floor_ratio'] = df_all['public_transport_station_km']/df_all['max_floor'].astype(float) #apartments near public t?
    #f_all['room_size'] = df_all['life_sq'] / df_all['num_room'].astype(float)
    df_all['gender_ratio'] = df_all['male_f']/df_all['female_f'].astype(float)
    df_all['kg_park_ratio'] = df_all['kindergarten_km']/df_all['park_km'].astype(float)
    df_all['high_ed_extent'] = df_all['school_km'] / df_all['kindergarten_km']
    df_all['pts_x_state'] = df_all['public_transport_station_km'] * df_all['state'].astype(float) #public trans * state of listing
    df_all['lifesq_x_state'] = df_all['life_sq'] * df_all['state'].astype(float)
    df_all['floor_x_state'] = df_all['floor'] * df_all['state'].astype(float)
    
    train['building_name'] = pd.factorize(train.sub_area + train['metro_km_avto'].astype(str))[0]
    test['building_name'] = pd.factorize(test.sub_area + test['metro_km_avto'].astype(str))[0]
    
    #########  END 2ND SET OF BILL S CHANGES
    
    
    
    def add_time_features(col):
       col_month_year = pd.Series(pd.factorize(train[col].astype(str) + month_year.astype(str))[0])
       train[col + '_month_year_cnt'] = col_month_year.map(col_month_year.value_counts())
    
       col_week_year = pd.Series(pd.factorize(train[col].astype(str) + week_year.astype(str))[0])
       train[col + '_week_year_cnt'] = col_week_year.map(col_week_year.value_counts())
    
    add_time_features('building_name')
    add_time_features('sub_area')
    
    def add_time_features(col):
       col_month_year = pd.Series(pd.factorize(test[col].astype(str) + month_year.astype(str))[0])
       test[col + '_month_year_cnt'] = col_month_year.map(col_month_year.value_counts())
    
       col_week_year = pd.Series(pd.factorize(test[col].astype(str) + week_year.astype(str))[0])
       test[col + '_week_year_cnt'] = col_week_year.map(col_week_year.value_counts())
    
    add_time_features('building_name')
    add_time_features('sub_area')
    
    
    # Remove timestamp column (may overfit the model in train)
    df_all.drop(['timestamp', 'timestamp_macro'], axis=1, inplace=True)
    
    
    factorize = lambda t: pd.factorize(t[1])[0]
    
    df_obj = df_all.select_dtypes(include=['object'])
    
    X_all = np.c_[
        df_all.select_dtypes(exclude=['object']).values,
        np.array(list(map(factorize, df_obj.iteritems()))).T
    ]
    print(X_all.shape)
    
    X_train = X_all[:num_train]
    X_test = X_all[num_train:]
    
    
    # Deal with categorical values
    df_numeric = df_all.select_dtypes(exclude=['object'])
    df_obj = df_all.select_dtypes(include=['object']).copy()
    
    for c in df_obj:
        df_obj[c] = pd.factorize(df_obj[c])[0]
    
    df_values = pd.concat([df_numeric, df_obj], axis=1)
    
    
    # Convert to numpy values
    X_all = df_values.values
    print(X_all.shape)
    
    X_train = X_all[:num_train]
    X_test = X_all[num_train:]
    
    df_columns = df_values.columns
    
    
    xgb_params = {
        'eta': 0.05,
        'max_depth': 5,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1
    }
    
    dtrain = xgb.DMatrix(X_train, y_train, feature_names=df_columns)
    dtest = xgb.DMatrix(X_test, feature_names=df_columns)
    
    num_boost_rounds = 420  # From Bruno's original CV, I think
    model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)
    
    y_pred = model.predict(dtest)
    
    df_sub = pd.DataFrame({'id': id_test, 'price_doc': y_pred})
    
    first_result = output.merge(df_sub, on="id", suffixes=['_louis','_bruno'])
    first_result["price_doc"] = np.exp( .714*np.log(first_result.price_doc_louis) +
                                        .286*np.log(first_result.price_doc_bruno) ) 
    
    first_result.to_csv('first_result.csv', index=False)
    
    print("model 2 done.")
    
    return first_result

stack_res = stack()
#stack_res = pd.read_csv('stack_2.csv')#gbr+xgbm
first_res = model2()

result = first_res.merge(stack_res, on="id", suffixes=['_follow','_stack'])
result["price_doc"] = np.exp( .78*np.log(result.price_doc_follow) +
                              .22*np.log(result.price_doc_stack) )
                              
result["price_doc"] =result["price_doc"] *0.9915      

result.drop(["price_doc_louis","price_doc_bruno","price_doc_follow","price_doc_stack"],axis=1,inplace=True)
result.to_csv('unadjusted_result.csv', index=False)



# APPLY PROBABILISTIC IMPROVEMENTS
preds = result
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Select investment sales from training set and generate frequency distribution
invest = train[train.product_type=="Investment"]
freqs = invest.price_doc.value_counts().sort_index()

# Select investment sales from test set predictions
test_invest_ids = test[test.product_type=="Investment"]["id"]
invest_preds = pd.DataFrame(test_invest_ids).merge(preds, on="id")

# Express X-axis of training set frequency distribution as logarithms, 
#    and save standard deviation to help adjust frequencies for time trend.
lnp = np.log(invest.price_doc)
stderr = lnp.std()
lfreqs = lnp.value_counts().sort_index()
	
# Adjust frequencies for time trend
lnp_diff = train_test_logmean_diff
lnp_mean = lnp.mean()
lnp_newmean = lnp_mean + lnp_diff


def norm_diff(value):
    return norm.pdf((value-lnp_diff)/stderr) / norm.pdf(value/stderr)

newfreqs = lfreqs * (pd.Series(lfreqs.index.values-lnp_newmean).apply(norm_diff).values)

# Logs of model-predicted prices
lnpred = np.log(invest_preds.price_doc)

# Create assumed probability distributions
stderr = prediction_stderr
mat =(np.array(newfreqs.index.values)[:,np.newaxis] - np.array(lnpred)[np.newaxis,:])/stderr
modelprobs = norm.pdf(mat)

# Multiply by frequency distribution.
freqprobs = pd.DataFrame( np.multiply( np.transpose(modelprobs), newfreqs.values ) )
freqprobs.index = invest_preds.price_doc.values
freqprobs.columns = freqs.index.values.tolist()

# Find mode for each case.
prices = freqprobs.idxmax(axis=1)

# Apply threshold to exclude low-confidence cases from recoding
priceprobs = freqprobs.max(axis=1)
mask = priceprobs<probthresh
prices[mask] = np.round(prices[mask].index,-rounder)

# Data frame with new predicitons
newpricedf = pd.DataFrame( {"id":test_invest_ids.values, "price_doc":prices} )

# Merge these new predictions (for just investment properties) back into the full prediction set.
newpreds = preds.merge(newpricedf, on="id", how="left", suffixes=("_old",""))
newpreds.loc[newpreds.price_doc.isnull(),"price_doc"] = newpreds.price_doc_old
newpreds.drop("price_doc_old",axis=1,inplace=True)
newpreds.head()

newpreds.to_csv('adjusted_result.csv', index=False)