# -*- coding: utf-8 -*-

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#from scipy.stats import spearmanr
from sklearn import preprocessing
#import knn_impute_sklearn 

def clean_data(train, test):
    #clean data
    
    #y_pre = train["price_doc"].copy(deep=True).values
    #print("y_pre shape"+str(len(y_pre)))
    
    bad_index = train[train.life_sq > train.full_sq].index
    train.ix[bad_index, "life_sq"] = np.NaN
    bad_index = test[test.life_sq > test.full_sq].index
    test.ix[bad_index, "life_sq"] = np.NaN
    
    equal_index = [601,1896,2791]
    test.ix[equal_index, "life_sq"] = test.ix[equal_index, "full_sq"]
    
    bad_index = train[train.life_sq < 5].index
    train.ix[bad_index, "life_sq"] = np.NaN
    bad_index = test[test.life_sq < 5].index
    test.ix[bad_index, "life_sq"] = np.NaN
    bad_index = train[train.full_sq < 5].index
    train.ix[bad_index, "full_sq"] = np.NaN
    bad_index = test[test.full_sq < 5].index
    test.ix[bad_index, "full_sq"] = np.NaN
    
    kitch_is_build_year = [13117]
    train.ix[kitch_is_build_year, "build_year"] = train.ix[kitch_is_build_year, "kitch_sq"]
    
    bad_index = train[train.kitch_sq >= train.life_sq].index
    train.ix[bad_index, "kitch_sq"] = np.NaN
    bad_index = test[test.kitch_sq >= test.life_sq].index
    test.ix[bad_index, "kitch_sq"] = np.NaN
    
    bad_index = train[(train.kitch_sq == 0).values + (train.kitch_sq == 1).values].index
    train.ix[bad_index, "kitch_sq"] = np.NaN
    bad_index = test[(test.kitch_sq == 0).values + (test.kitch_sq == 1).values].index
    test.ix[bad_index, "kitch_sq"] = np.NaN
    
    bad_index = train[(train.full_sq > 210) & (train.life_sq / train.full_sq < 0.3)].index
    train.ix[bad_index, "full_sq"] = np.NaN
    bad_index = test[(test.full_sq > 150) & (test.life_sq / test.full_sq < 0.3)].index
    test.ix[bad_index, "full_sq"] = np.NaN
    
    bad_index = train[train.life_sq > 300].index
    train.ix[bad_index, ["life_sq", "full_sq"]] = np.NaN
    bad_index = test[test.life_sq > 200].index
    test.ix[bad_index, ["life_sq", "full_sq"]] = np.NaN
    
    train.product_type.value_counts(normalize= True)
    test.product_type.value_counts(normalize= True)
    
    bad_index = train[train.build_year < 1500].index
    train.ix[bad_index, "build_year"] = np.NaN
    bad_index = test[test.build_year < 1500].index
    test.ix[bad_index, "build_year"] = np.NaN
    
    bad_index = train[train.num_room == 0].index 
    train.ix[bad_index, "num_room"] = np.NaN
    bad_index = test[test.num_room == 0].index 
    test.ix[bad_index, "num_room"] = np.NaN
    
    bad_index = [10076, 11621, 17764, 19390, 24007, 26713, 29172]
    train.ix[bad_index, "num_room"] = np.NaN
    bad_index = [3174, 7313]
    test.ix[bad_index, "num_room"] = np.NaN
    
    #bad_index = train[(train.floor == 0).values * (train.max_floor == 0).values].index
    #train.ix[bad_index, ["max_floor", "floor"]] = np.NaN
    
    bad_index = train[train.floor == 0].index
    train.ix[bad_index, "floor"] = np.NaN
    bad_index = train[train.max_floor == 0].index
    train.ix[bad_index, "max_floor"] = np.NaN
    
    bad_index = test[test.floor == 0].index
    test.ix[bad_index, "floor"] = np.NaN
    bad_index = test[test.max_floor == 0].index
    test.ix[bad_index, "max_floor"] = np.NaN
    
    bad_index = train[train.floor > train.max_floor].index
    train.ix[bad_index, "max_floor"] = np.NaN
    bad_index = test[test.floor > test.max_floor].index
    test.ix[bad_index, "max_floor"] = np.NaN
    
    #train.floor.describe(percentiles= [0.9999])
    bad_index = [23584]
    train.ix[bad_index, "floor"] = np.NaN
    
    #train.material.value_counts()
    #test.material.value_counts()
    #train.state.value_counts()
    bad_index = train[train.state == 33].index
    train.ix[bad_index, "state"] = np.NaN
    #test.state.value_counts()
    
    # brings error down a lot by removing extreme price per sqm
    train.loc[train.full_sq == 0, 'full_sq'] = 50
    train = train[train.price_doc/train.full_sq <= 600000]
    train = train[train.price_doc/train.full_sq >= 10000]

    #product_type
    train.loc[train['product_type'].isnull(),'product_type']='Investment'
    test.loc[test['product_type'].isnull(),'product_type']='Investment'    
    
    #y_post = train["price_doc"].copy(deep=True).values
    #print("y_post shape"+str(len(y_post)))
    #print(sum(y_pre==y_post))
    
    print("clean data done.")
    
    return train, test

def handle_missing_value_mostfreq(train, test):
    #num_train = len(train)
    #df_all = pd.concat([train, test])     
    
    train = train.replace([np.inf, -np.inf], np.nan)
    test = test.replace([np.inf, -np.inf], np.nan)

    #product_type
    #train.loc[train['product_type'].isnull(),'product_type']='Investment'
    #test.loc[test['product_type'].isnull(),'product_type']='Investment' 
    
    # Create a list of columns that have missing values and an index (True / False)
    train_missing = train.isnull().sum(axis = 0).reset_index()
    train_missing.columns = ['column_name', 'missing_count']
    idx_ = train_missing['missing_count'] > 0
    train_missing = train_missing.ix[idx_]
    cols_missing = train_missing.column_name.values
    idx_cols_missing = train.columns.isin(cols_missing)
    
    print("train missing columns:" + str(len(cols_missing)))
    
    # Instantiate an imputer
    imputer = preprocessing.Imputer(missing_values='NaN', strategy = 'median', axis = 0)   
    #imputer = knn_impute_sklearn.Imputer(missing_values='NaN', strategy = 'knn', axis = 0)   

    # Fit the imputer using all of our data (but not any dates)
    #imputer.fit(train.ix[:, idx_cols_missing])  
    #imputer.fit(train[:, idx_cols_missing]) 
    # Apply the imputer
    #train.ix[:, idx_cols_missing] = imputer.transform(train.ix[:, idx_cols_missing])
    #train[:, idx_cols_missing] = imputer.transform(train[:, idx_cols_missing])
    train.ix[:, idx_cols_missing] = imputer.fit_transform(train.ix[:, idx_cols_missing])
    
    train_missing = train.isnull().sum(axis = 0).reset_index()
    train_missing.columns = ['column_name', 'missing_count']
    idx_ = train_missing['missing_count'] > 0
    train_missing = train_missing.ix[idx_]
    cols_missing = train_missing.column_name.values
    idx_cols_missing = train.columns.isin(cols_missing)
    
    print("train missing columns:" + str(len(cols_missing)))
    
    # Create a list of columns that have missing values and an index (True / False)
    test_missing = test.isnull().sum(axis = 0).reset_index()
    test_missing.columns = ['column_name', 'missing_count']
    idx_ = test_missing['missing_count'] > 0
    test_missing = test_missing.ix[idx_]
    cols_missing = test_missing.column_name.values
    idx_cols_missing = test.columns.isin(cols_missing)
    
    print("test missing columns:" + str(len(cols_missing)))
        
    # Fit the imputer using all of our data (but not any dates)
    #imputer.fit(test.ix[:, idx_cols_missing])   
    #imputer.fit(test[:, idx_cols_missing])
    # Apply the imputer
    #test.ix[:, idx_cols_missing] = imputer.transform(test.ix[:, idx_cols_missing])
    #test[:, idx_cols_missing] = imputer.transform(test[:, idx_cols_missing])
    test.ix[:, idx_cols_missing] = imputer.fit_transform(test.ix[:, idx_cols_missing])
    
    test_missing = test.isnull().sum(axis = 0).reset_index()
    test_missing.columns = ['column_name', 'missing_count']
    idx_ = test_missing['missing_count'] > 0
    test_missing = test_missing.ix[idx_]
    cols_missing = test_missing.column_name.values
    idx_cols_missing = test.columns.isin(cols_missing)
    
    print("test missing columns:" + str(len(cols_missing)))
    
    
    #print("test missing columns:" + str(len(idx_cols_missing)))
    
    print("handle_missing_value_mostfreq done.")
    
    return train, test

    
def handle_missing_value_subarea(train, test):
    # handling missing value by taking statistics from local area
    train_sub_area = train.sub_area.unique()
    test_sub_area = test.sub_area.unique()
    
    #full_sq
    for i in train_sub_area:
        if train[(train['full_sq'].isnull()) & (train['sub_area']==i)].shape[0]>0:
            train.loc[(train['full_sq'].isnull()) & (train['sub_area']==i), 'full_sq']=train.loc[(train['full_sq'].notnull()) & (train['sub_area']==i), 'full_sq'].mode().values[0]
    for i in test_sub_area:
        if test[(test['full_sq'].isnull()) & (test['sub_area']==i)].shape[0]>0:
            test.loc[(test['full_sq'].isnull()) & (test['sub_area']==i), 'full_sq']=test.loc[(test['full_sq'].notnull()) & (test['sub_area']==i), 'full_sq'].mode().values[0]
    
    #floor
    for i in train_sub_area:
        if train[(train['floor'].isnull()) & (train['sub_area']==i)].shape[0]>0:
            train.loc[(train['floor'].isnull()) & (train['sub_area']==i), 'floor']=train.loc[(train['floor'].notnull()) & (train['sub_area']==i), 'floor'].mode().values[0]
    for i in test_sub_area:
        if test[(test['floor'].isnull()) & (test['sub_area']==i)].shape[0]>0:
            test.loc[(test['floor'].isnull()) & (test['sub_area']==i), 'floor']=test.loc[(test['floor'].notnull()) & (test['sub_area']==i), 'floor'].mode().values[0]
    
    #max floor
    for i in train_sub_area:
        if train[(train['max_floor'].isnull()) & (train['sub_area']==i)].shape[0]>0:
            if i=='Poselenie Shhapovskoe':
                train.loc[(train['max_floor'].isnull()) & (train['sub_area']==i),'max_floor']=train.loc[(train['max_floor'].notnull()) & (train['sub_area']==i),'max_floor'].median()
            else:
                train.loc[(train['max_floor'].isnull()) & (train['sub_area']==i),'max_floor']=train.loc[(train['max_floor'].notnull()) & (train['sub_area']==i),'max_floor'].median()
    for i in test_sub_area:
        if test[(test['max_floor'].isnull()) & (test['sub_area']==i)].shape[0]>0:
            if i=='Poselenie Shhapovskoe':
                test.loc[(test['max_floor'].isnull()) & (test['sub_area']==i),'max_floor']=test.loc[(test['max_floor'].notnull()) & (test['sub_area']==i),'max_floor'].median()
            else:
                test.loc[(test['max_floor'].isnull()) & (test['sub_area']==i),'max_floor']=test.loc[(test['max_floor'].notnull()) & (test['sub_area']==i),'max_floor'].median()
    
    #materials
    for i in train_sub_area:
        if train[(train['material'].isnull()) & (train['sub_area']==i)].shape[0]>0:
            train.loc[(train['material'].isnull()) & (train['sub_area']==i),'material']=train.loc[(train['material'].notnull()) & (train['sub_area']==i),'material'].median()
    for i in test_sub_area:
        if test[(test['material'].isnull()) & (test['sub_area']==i)].shape[0]>0:
            test.loc[(test['material'].isnull()) & (test['sub_area']==i),'material']=test.loc[(test['material'].notnull()) & (test['sub_area']==i),'material'].median()
    
    #build year
    for i in train_sub_area:
        if train[(train['build_year'].isnull()) & (train['sub_area']==i)].shape[0]>0:
            if i=='Poselenie Voronovskoe':
                train.loc[(train['build_year'].isnull()) & (train['sub_area']==i),'build_year']=2014
            elif i=='Poselenie Shhapovskoe':
                train.loc[(train['build_year'].isnull()) & (train['sub_area']==i),'build_year']=2011
            else:
                train.loc[(train['build_year'].isnull()) & (train['sub_area']==i),'build_year']=train.loc[(train['build_year'].notnull()) & (train['sub_area']==i),'build_year'].median()
    for i in test_sub_area:
        if test[(test['build_year'].isnull()) & (test['sub_area']==i)].shape[0]>0:
            if i=='Poselenie Voronovskoe':
                test.loc[(test['build_year'].isnull()) & (test['sub_area']==i),'build_year']=2014
            elif i=='Poselenie Shhapovskoe':
                test.loc[(test['build_year'].isnull()) & (test['sub_area']==i),'build_year']=2011
            else:
                test.loc[(test['build_year'].isnull()) & (test['sub_area']==i),'build_year']=test.loc[(test['build_year'].notnull()) & (test['sub_area']==i),'build_year'].median()
    
    #state
    for i in train_sub_area:
        if train[(train['state'].isnull()) & (train['sub_area']==i)].shape[0]>0:
            if (i=='Poselenie Klenovskoe' or i=='Poselenie Kievskij'):
                train.loc[(train['state'].isnull()) & (train['sub_area']==i),'state']=2
            else:
                train.loc[(train['state'].isnull()) & (train['sub_area']==i),'state']=train.loc[(train['state'].notnull()) & (train['sub_area']==i),'state'].median()
    for i in test_sub_area:
        if test[(test['state'].isnull()) & (test['sub_area']==i)].shape[0]>0:
            if (i=='Poselenie Klenovskoe' or i=='Poselenie Kievskij'):
                test.loc[(test['state'].isnull()) & (test['sub_area']==i),'state']=2
            else:
                test.loc[(test['state'].isnull()) & (test['sub_area']==i),'state']=test.loc[(test['state'].notnull()) & (test['sub_area']==i),'state'].median()
    
    #doing things a bit differnt for rooms, base it off sq ft.
    x=train.loc[(train.full_sq.notnull())&(train.num_room.notnull()),'full_sq']
    y=train.loc[(train.full_sq.notnull())&(train.num_room.notnull()),'num_room']
    rooms=np.polyfit(x,y,1)
    train.loc[train['num_room'].isnull(),'num_room']=round(train.loc[train['num_room'].isnull(),'full_sq']*rooms[0]+rooms[1])
    
    x=test.loc[(test.full_sq.notnull())&(test.num_room.notnull()),'full_sq']
    y=test.loc[(test.full_sq.notnull())&(test.num_room.notnull()),'num_room']
    rooms=np.polyfit(x,y,1)
    test.loc[test['num_room'].isnull(),'num_room']=round(test.loc[test['num_room'].isnull(),'full_sq']*rooms[0]+rooms[1])
    
    #kitchen space will be same
    x=train.loc[(train.kitch_sq.notnull())&(train.kitch_sq.notnull()),'full_sq']
    y=train.loc[(train.kitch_sq.notnull())&(train.kitch_sq.notnull()),'kitch_sq']
    kitch=np.polyfit(x,y,1)
    train.loc[train['kitch_sq'].isnull(),'kitch_sq']=round(train.loc[train['kitch_sq'].isnull(),'full_sq']*kitch[0]+kitch[1])
    
    x=test.loc[(test.kitch_sq.notnull())&(test.kitch_sq.notnull()),'full_sq']
    y=test.loc[(test.kitch_sq.notnull())&(test.kitch_sq.notnull()),'kitch_sq']
    kitch=np.polyfit(x,y,1)
    test.loc[test['kitch_sq'].isnull(),'kitch_sq']=round(test.loc[test['kitch_sq'].isnull(),'full_sq']*kitch[0]+kitch[1])
    
    #and fix up the life-sq
    x=train.loc[(train.full_sq.notnull())&(train.life_sq.notnull()),'full_sq']
    y=train.loc[(train.full_sq.notnull())&(train.life_sq.notnull()),'life_sq']
    life=np.polyfit(x,y,1)
    train.loc[train['life_sq'].isnull(),'life_sq']=round(train.loc[train['life_sq'].isnull(),'full_sq']*life[0]+life[1])
    
    x=test.loc[(test.full_sq.notnull())&(test.life_sq.notnull()),'full_sq']
    y=test.loc[(test.full_sq.notnull())&(test.life_sq.notnull()),'life_sq']
    life=np.polyfit(x,y,1)
    test.loc[test['life_sq'].isnull(),'life_sq']=round(test.loc[test['life_sq'].isnull(),'full_sq']*life[0]+life[1])
    
    i='children_school'
    j='school_quota'
    x=train.loc[(train[i].notnull())&(train[j].notnull()),i]
    y=train.loc[(train[i].notnull())&(train[j].notnull()),j]
    fit=np.polyfit(x,y,1)
    train.loc[train[j].isnull(),j]=round(train.loc[train[j].isnull(),i]*fit[0]+fit[1])
    
    print("handle_missing_value_subarea done.")
    
    return train, test

def add_features(train):
    '''
    train["year"]  = train["timestamp"].dt.year 
    month_year = (train.timestamp.dt.month*30 + train.timestamp.dt.year * 365)
    month_year_cnt_map = month_year.value_counts().to_dict()
    train['month_year_cnt'] = month_year.map(month_year_cnt_map)
    
    # Add week-year count
    week_year = (train.timestamp.dt.weekofyear*7 + train.timestamp.dt.year * 365)
    week_year_cnt_map = week_year.value_counts().to_dict()
    train['week_year_cnt'] = week_year.map(week_year_cnt_map)
    
    # Add month and day-of-week
    train['month'] = train.timestamp.dt.month
    train['dow'] = train.timestamp.dt.dayofweek'''
    
    # Other feature engineering
    train.loc[:,'rel_floor'] = 0.05+train.loc[:,'floor'] / train.loc[:,'max_floor'].astype(float)
    train.loc[:,'rel_kitch_sq'] = 0.05+train.loc[:,'kitch_sq'] / train.loc[:,'full_sq'].astype(float)
    
    #train.apartment_name=train.sub_area + train['metro_km_avto'].astype(str)
    
    train.loc[:,'room_size'] = train.loc[:,'life_sq'] / train.loc[:,'num_room'].astype(float)
    
    train.loc[:,'extra_sq'] = train.loc[:,'full_sq'] - train.loc[:,'life_sq']
    
    #train['age_of_building'] = train['year'] - train['build_year']
    
    #train['area_per_room'] = train['life_sq'] / train['num_room'].astype(float) #rough area per room
    train.loc[:,'livArea_ratio'] = train.loc[:,'life_sq'] / train.loc[:,'full_sq'].astype(float) #rough living area
    train.loc[:,'yrs_old'] = 2017 - train.loc[:,'build_year'].astype(float) #years old from 2017
    train.loc[:,'avgfloor_sq'] = train.loc[:,'life_sq']/train.loc[:,'max_floor'].astype(float) #living area per floor
    train.loc[:,'pts_floor_ratio'] = train.loc[:,'public_transport_station_km']/train.loc[:,'max_floor'].astype(float)
    
    # doubled a var by accident
    train.loc[:,'gender_ratio'] = train.loc[:,'male_f']/train.loc[:,'female_f'].astype(float)
    train.loc[:,'kg_park_ratio'] = train.loc[:,'kindergarten_km']/train.loc[:,'park_km'].astype(float) #significance of children?
    train.loc[:,'high_ed_extent'] = train.loc[:,'school_km'] / train.loc[:,'kindergarten_km'] #schooling
    train.loc[:,'pts_x_state'] = train.loc[:,'public_transport_station_km'] * train.loc[:,'state'].astype(float) #public trans * state of listing
    train.loc[:,'lifesq_x_state'] = train.loc[:,'life_sq'] * train.loc[:,'state'].astype(float) #life_sq times the state of the place
    train.loc[:,'floor_x_state'] = train.loc[:,'floor'] * train.loc[:,'state'].astype(float) #relative floor * the state of the place
    
    # district dimension
    train.loc[:,'dimension_sub_area'] = train.loc[:,'area_m'].apply(np.sqrt, 0)
    
    # high school -> school_quota / 7to14_age_Persons
    train.loc[:,'school_seat_availability'] = train.loc[:,'school_quota']/train.loc[:,'children_school']
    
    # number of school per area
    train.loc[:,'school_per_area'] = 1e7 * train.loc[:,'school_education_centers_raion'] / train.loc[:,'area_m']
    
    # how closeby school
    train.loc[:,'school_closeness'] = train.loc[:,'school_km'] / train.loc[:,'dimension_sub_area']
    # Preschool seat per child
    train.loc[:,'preschool_seat_availability'] = train.loc[:,'preschool_quota'] / train.loc[:,'children_preschool']
    
    # number of preschool per area
    train.loc[:,'preschool_per_area']  = 1e7 * train.loc[:,'preschool_education_centers_raion'] / train.loc[:,'area_m']
    
    # how close is preschool
    train.loc[:,'preschool_closeness'] = train.loc[:,'preschool_km'] / train.loc[:,'dimension_sub_area']
    
    # is preschool same as school
    train.loc[:,'diff_school'] = train.loc[:,'preschool_km'] == train.loc[:,'school_km']
    
    # closeness of offices 
    train.loc[:,'close_office'] = train.loc[:,'office_km'] / train.loc[:,'dimension_sub_area']
    
    # work_availability
    train.loc[:,'work_avail'] = train.loc[:,'office_raion'] / train.loc[:,'work_all']
    
    # density of healthcare centres
    train.loc[:,'healthcare_density'] = 1e7 * train.loc[:,'healthcare_centers_raion'] / train.loc[:,'area_m']
    
    # Pollution coeff - relative dist. to indu_zone
    train.loc[:,'safe_nature'] = train.loc[:,'industrial_km'] / train.loc[:,'green_zone_km']
   
    # Pollution coeff - relative dist. to water treatment
    train.loc[:,'safe_watre'] = train.loc[:,'industrial_km'] / train.loc[:,'water_treatment_km']
   
    # closeness of public healthcare
    train.loc[:,'close_public_health'] = train.loc[:,'public_healthcare_km'] / train.loc[:,'dimension_sub_area']
    
    # close to office?
    train.loc[:,'close_office'] = train.loc[:,'office_km'] / train.loc[:,'dimension_sub_area']
    
    # Density of shopping malls
    train.loc[:,'shop_density'] = 1e7 * train.loc[:,'shopping_centers_raion'] / train.loc[:,'area_m']
    
    # closeness of shopping malls 
    train.loc[:,'close_shops'] =  train.loc[:,'shopping_centers_km'] / train.loc[:,'dimension_sub_area']
    
    # New City or Old city
    train.loc[:,'build_cound_before_1995'] = train.loc[:,'build_count_before_1920'] + train.loc[:,'build_count_1921-1945'] +train.loc[:,'build_count_1946-1970']+train.loc[:,'build_count_1971-1995']
    
    train.loc[:,'new_or_old_city'] =  train.loc[:,'build_count_after_1995'] / (train.loc[:,'build_cound_before_1995'])

    return train
    
    

def add_features_both(train, test):
    #add year
    #y_pre = train["price_doc"].copy(deep=True).values
    #print("y_pre shape"+str(len(y_pre)))    
    
    #train["timestamp"] = pd.to_datetime(train["timestamp"])
    train["year"]  = train["timestamp"].dt.year    
    #test["timestamp"] = pd.to_datetime(test["timestamp"])
    test["year"]  = test["timestamp"].dt.year
    
    '''# Add month-year
    month_year = (train.timestamp.dt.month + train.timestamp.dt.year * 100)
    month_year_cnt_map = month_year.value_counts().to_dict()
    train['month_year_cnt'] = month_year.map(month_year_cnt_map)
    
    month_year = (test.timestamp.dt.month + test.timestamp.dt.year * 100)
    month_year_cnt_map = month_year.value_counts().to_dict()
    test['month_year_cnt'] = month_year.map(month_year_cnt_map)
    
    # Add week-year count
    week_year = (train.timestamp.dt.weekofyear + train.timestamp.dt.year * 100)
    week_year_cnt_map = week_year.value_counts().to_dict()
    train['week_year_cnt'] = week_year.map(week_year_cnt_map)
    
    week_year = (test.timestamp.dt.weekofyear + test.timestamp.dt.year * 100)
    week_year_cnt_map = week_year.value_counts().to_dict()
    test['week_year_cnt'] = week_year.map(week_year_cnt_map)'''
    # Add month-year
    month_year = (train.timestamp.dt.month*30 + train.timestamp.dt.year * 365)
    month_year_cnt_map = month_year.value_counts().to_dict()
    train['month_year_cnt'] = month_year.map(month_year_cnt_map)
    
    month_year = (test.timestamp.dt.month*30 + test.timestamp.dt.year * 365)
    month_year_cnt_map = month_year.value_counts().to_dict()
    test['month_year_cnt'] = month_year.map(month_year_cnt_map)
    
    # Add week-year count
    week_year = (train.timestamp.dt.weekofyear*7 + train.timestamp.dt.year * 365)
    week_year_cnt_map = week_year.value_counts().to_dict()
    train['week_year_cnt'] = week_year.map(week_year_cnt_map)
    
    week_year = (test.timestamp.dt.weekofyear*7 + test.timestamp.dt.year * 365)
    week_year_cnt_map = week_year.value_counts().to_dict()
    test['week_year_cnt'] = week_year.map(week_year_cnt_map)
    
    # Add month and day-of-week
    train['month'] = train.timestamp.dt.month
    train['dow'] = train.timestamp.dt.dayofweek
    
    test['month'] = test.timestamp.dt.month
    test['dow'] = test.timestamp.dt.dayofweek
    
    # Other feature engineering
    train['rel_floor'] = 0.05+train['floor'] / train['max_floor'].astype(float)
    train['rel_kitch_sq'] = 0.05+train['kitch_sq'] / train['full_sq'].astype(float)
    
    test['rel_floor'] = 0.05+test['floor'] / test['max_floor'].astype(float)
    test['rel_kitch_sq'] = 0.05+test['kitch_sq'] / test['full_sq'].astype(float)
    
    train.apartment_name=train.sub_area + train['metro_km_avto'].astype(str)
    test.apartment_name=test.sub_area + train['metro_km_avto'].astype(str)
    
    train['room_size'] = train['life_sq'] / train['num_room'].astype(float)
    test['room_size'] = test['life_sq'] / test['num_room'].astype(float)
    
    train['extra_sq'] = train['full_sq'] - train['life_sq']
    test['extra_sq'] = test['full_sq'] - test['life_sq']
    
    train['age_of_building'] = train['year'] - train['build_year']
    test['age_of_building'] = test['year'] - test['build_year']
    
    #train['area_per_room'] = train['life_sq'] / train['num_room'].astype(float) #rough area per room
    train['livArea_ratio'] = train['life_sq'] / train['full_sq'].astype(float) #rough living area
    train['yrs_old'] = 2017 - train['build_year'].astype(float) #years old from 2017
    train['avgfloor_sq'] = train['life_sq']/train['max_floor'].astype(float) #living area per floor
    train['pts_floor_ratio'] = train['public_transport_station_km']/train['max_floor'].astype(float)
    # looking for significance of apartment buildings near public t 
    #train['room_size'] = train['life_sq'] / train['num_room'].astype(float)
    # doubled a var by accident
    train['gender_ratio'] = train['male_f']/train['female_f'].astype(float)
    train['kg_park_ratio'] = train['kindergarten_km']/train['park_km'].astype(float) #significance of children?
    train['high_ed_extent'] = train['school_km'] / train['kindergarten_km'] #schooling
    train['pts_x_state'] = train['public_transport_station_km'] * train['state'].astype(float) #public trans * state of listing
    train['lifesq_x_state'] = train['life_sq'] * train['state'].astype(float) #life_sq times the state of the place
    train['floor_x_state'] = train['floor'] * train['state'].astype(float) #relative floor * the state of the place
    
    #test['area_per_room'] = test['life_sq'] / test['num_room'].astype(float)
    test['livArea_ratio'] = test['life_sq'] / test['full_sq'].astype(float)
    test['yrs_old'] = 2017 - test['build_year'].astype(float)
    test['avgfloor_sq'] = test['life_sq']/test['max_floor'].astype(float) #living area per floor
    test['pts_floor_ratio'] = test['public_transport_station_km']/test['max_floor'].astype(float) #apartments near public t?
    #test['room_size'] = test['life_sq'] / test['num_room'].astype(float)
    test['gender_ratio'] = test['male_f']/test['female_f'].astype(float)
    test['kg_park_ratio'] = test['kindergarten_km']/test['park_km'].astype(float)
    test['high_ed_extent'] = test['school_km'] / test['kindergarten_km']
    test['pts_x_state'] = test['public_transport_station_km'] * test['state'].astype(float) #public trans * state of listing
    test['lifesq_x_state'] = test['life_sq'] * test['state'].astype(float)
    test['floor_x_state'] = test['floor'] * test['state'].astype(float)
    
    # district dimension
    train['dimension_sub_area'] = train['area_m'].apply(np.sqrt, 0)
    test['dimension_sub_area'] = test['area_m'].apply(np.sqrt, 0)
    
    # high school -> school_quota / 7to14_age_Persons
    train['school_seat_availability'] = train['school_quota']/train['children_school']
    test['school_seat_availability'] = test['school_quota']/test['children_school']
    
    # number of school per area
    train['school_per_area'] = 1e7 * train['school_education_centers_raion'] / train['area_m']
    test['school_per_area'] = 1e7 * test['school_education_centers_raion'] / test['area_m']
    
    # how closeby school
    train['school_closeness'] = train['school_km'] / train['dimension_sub_area']
    test['school_closeness'] = test['school_km'] / test['dimension_sub_area']
    
    # Preschool seat per child
    train['preschool_seat_availability'] = train['preschool_quota'] / train['children_preschool']
    test['preschool_seat_availability'] = test['preschool_quota'] / test['children_preschool']
    
    # number of preschool per area
    train['preschool_per_area']  = 1e7 * train['preschool_education_centers_raion'] / train['area_m']
    test['preschool_per_area']  = 1e7 * test['preschool_education_centers_raion'] / test['area_m']
    
    # how close is preschool
    train['preschool_closeness'] = train['preschool_km'] / train['dimension_sub_area']
    test['preschool_closeness'] = test['preschool_km'] / test['dimension_sub_area']
    
    # is preschool same as school
    train['diff_school'] = train['preschool_km'] == train['school_km']
    test['diff_school'] = test['preschool_km'] == test['school_km']
    
    # closeness of offices 
    train['close_office'] = train['office_km'] / train['dimension_sub_area']
    test['close_office'] = test['office_km'] / test['dimension_sub_area']
    
    # work_availability
    train['work_avail'] = train['office_raion'] / train['work_all']
    test['work_avail'] = test['office_raion'] / test['work_all']
    
    # density of healthcare centres
    train['healthcare_density'] = 1e7 * train['healthcare_centers_raion'] / train['area_m']
    test['healthcare_density'] = 1e7 * test['healthcare_centers_raion'] / test['area_m']
    
    # Pollution coeff - relative dist. to indu_zone
    train['safe_nature'] = train['industrial_km'] / train['green_zone_km']
    test['safe_nature'] = test['industrial_km'] / test['green_zone_km']
    
    # Pollution coeff - relative dist. to water treatment
    train['safe_watre'] = train['industrial_km'] / train['water_treatment_km']
    test['safe_watre'] = test['industrial_km'] / test['water_treatment_km']
    
    # closeness of public healthcare
    train['close_public_health'] = train['public_healthcare_km'] / train['dimension_sub_area']
    test['close_public_health'] = test['public_healthcare_km'] / test['dimension_sub_area']
    
    # close to office?
    train['close_office'] = train['office_km'] / train['dimension_sub_area']
    test['close_office'] = test['office_km'] / test['dimension_sub_area']
    
    # Density of shopping malls
    train['shop_density'] = 1e7 * train['shopping_centers_raion'] / train['area_m']
    test['shop_density'] = 1e7 * test['shopping_centers_raion'] / test['area_m']
    
    # closeness of shopping malls 
    train['close_shops'] =  train['shopping_centers_km'] / train['dimension_sub_area']
    test['close_shops'] =  test['shopping_centers_km'] / test['dimension_sub_area']
    
    # New City or Old city
    train['build_cound_before_1995'] = train['build_count_before_1920'] + train['build_count_1921-1945'] +train['build_count_1946-1970']+train['build_count_1971-1995']
    test['build_cound_before_1995'] = test['build_count_before_1920'] + test['build_count_1921-1945'] +test['build_count_1946-1970']+test['build_count_1971-1995']
    
    train['new_or_old_city'] =  train['build_count_after_1995'] / (train['build_cound_before_1995'])
    test['new_or_old_city'] =  test['build_count_after_1995'] / (test['build_cound_before_1995'])
    
    #y_post = train["price_doc"].copy(deep=True).values
    #print("y_post shape"+str(len(y_post)))
    #print(sum(y_pre==y_post))

    print("add_features done.")    
    return train, test

def cat_2_num(train, test):
    #encoding categorical features
    df_all = pd.concat([train, test])
    for c in df_all.columns:
        if df_all[c].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(df_all[c].values)) 
            df_all[c] = lbl.transform(list(df_all[c].values))
    
    num_train = len(train)
            
    train = df_all[:num_train]
    test = df_all[num_train:]
    
    return train, test

def add_decomposition_features(train, test):
    from sklearn.decomposition import PCA, TruncatedSVD, FastICA#NMF
    from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
    y = train['price_doc']
    time_train = train['timestamp']
    time_test = test['timestamp']
    
    train = train.drop(["timestamp", "price_doc"], axis=1)
    test = test.drop(["timestamp"], axis=1)
    
    train, test = handle_missing_value_mostfreq(train, test)
    
    train, test = cat_2_num(train, test)
    #x_train = train.drop(["price_doc"], axis=1)
    n_comp = 30
    
    #min_max_scaler = preprocessing.MinMaxScaler()
    #train = min_max_scaler.fit_transform(train)
    #test = min_max_scaler.fit_transform(test)
    
    #test_value = test.values
    #for i in np.argwhere(np.isnan(test_value)):
    #    print(test.ix[:,i])
    #print(np.argwhere(np.isnan(test_value)))
    
    
    # tSVD
    tsvd = TruncatedSVD(n_components=n_comp, random_state=420)
    tsvd_results_train = tsvd.fit_transform(train)
    tsvd_results_test = tsvd.transform(test)
    
    # PCA
    pca = PCA(n_components=n_comp, random_state=420)
    pca2_results_train = pca.fit_transform(train)
    pca2_results_test = pca.transform(test)
    
    # ICA
    ica = FastICA(n_components=n_comp, random_state=420)
    ica2_results_train = ica.fit_transform(train)
    ica2_results_test = ica.transform(test)
    
    # GRP
    grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)
    grp_results_train = grp.fit_transform(train)
    grp_results_test = grp.transform(test)
    
    # SRP
    srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)
    srp_results_train = srp.fit_transform(train)
    srp_results_test = srp.transform(test)
    
    #NMF
    #nmf = NMF(n_components = n_comp, random_state=420)
    
    # Append decomposition components to datasets
    for i in range(1, n_comp + 1):
        train['pca_' + str(i)] = pca2_results_train[:, i - 1]
        test['pca_' + str(i)] = pca2_results_test[:, i - 1]
    
        train['ica_' + str(i)] = ica2_results_train[:, i - 1]
        test['ica_' + str(i)] = ica2_results_test[:, i - 1]
    
        train['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
        test['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]
    
        train['grp_' + str(i)] = grp_results_train[:, i - 1]
        test['grp_' + str(i)] = grp_results_test[:, i - 1]
    
        train['srp_' + str(i)] = srp_results_train[:, i - 1]
        test['srp_' + str(i)] = srp_results_test[:, i - 1]
    
    train = pd.concat([train, y, time_train], axis=1)
    test = pd.concat([test, time_test], axis=1)
    print("add decomposition features done.")
    return train, test
    

def add_spearman_features(train, test):
    y = train['price_doc'].values
    y_mean = np.mean(y)
    
    X_ex = train.select_dtypes(exclude = ['float64', 'int64']).copy(deep=True)
    X_t_ex = test.select_dtypes(exclude = ['float64', 'int64']).copy(deep=True)
    #print(X_ex.shape)
    #print(X_t_ex.shape)
    
    #train.drop('y', axis=1, inplace=True)
    X = train.select_dtypes(include = ['float64', 'int64']).copy(deep=True)
    X_t = test.select_dtypes(include = ['float64', 'int64']).copy(deep=True)
    #print(X.shape)
    #print(X_t.shape)
    
    """
    #fill nan with mean 
    from sklearn.preprocessing import Imputer
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    X = imp.fit_transform(X)
    X_t = imp.fit_transform(X_t)
    
    all_cols = X.columns
    
    
    missing_df = X.isnull().sum(axis=0).reset_index()
    missing_df.columns = ['column_name', 'missing_count']
    missing_df = missing_df.ix[missing_df['missing_count']>0]
    print(missing_df)
    
    missing_df_t = X_t.isnull().sum(axis=0).reset_index()
    missing_df_t.columns = ['column_name', 'missing_count']
    missing_df_t = missing_df_t.ix[missing_df_t['missing_count']>0]
    print(missing_df_t)"""

    #spearman rank coefficient
    corr_spear = X.iloc[:, 1:].corr(method='spearman')
    cor_dict = corr_spear['price_doc'].to_dict()
    del cor_dict['price_doc']
    
    #merge features
    cafe_3 = []
    cafe_2_3 = []
    cafe_1_2 = []
    cafe_0_1 = []
    cafe_0 = []
    trc_count = []
    trc_sqm = []
    sport_count = []
    office_count = []
    office_sqm = []
    km_3 = []
    km_2_3 = []
    km_1_2 = []
    km_0_1 = []
    church_count = []
    region_2 = []
    region_0_2 = []
    region_0 = []
    leisure_count = []
    male_female_all = []
    prom_part = [] 
    prom_part_0 = []
    green_part = []
    market_count = []
    mosque_count = []
    for key, val in cor_dict.items():
        if "cafe_" in key:
            if val>=0.3:
                cafe_3.append(key)
            elif val>=0.2 and val<0.3:
                cafe_2_3.append(key)
            elif val>=0.1 and val<0.2:
                cafe_1_2.append(key)
            elif val>=0.0 and val<0.1:
                cafe_0_1.append(key)
            elif val<0.0:
                cafe_0.append(key)
        if "trc_count" in key:
            trc_count.append(key)
        if "trc_sqm" in key:
            trc_sqm.append(key)
        if "sport_count" in key:
            sport_count.append(key)
        if "office_count" in key:
            office_count.append(key)
        if "office_sqm" in key:
            office_sqm.append(key)
        if "_km" in key:
            if val<=-0.3:
                km_3.append(key)
            elif val<=-0.2 and val>-0.3:
                km_2_3.append(key)
            elif val<=-0.1 and val>-0.2:
                km_1_2.append(key)
            elif val<=0.0 and val>-0.1:
                km_0_1.append(key)
        if "church_count" in key:
            church_count.append(key)
        if "_raion" in key:
            if val>=0.2:
                region_2.append(key)
            elif val>=0.0 and val<0.2:
                region_0_2.append(key)
            elif val<0.0:
                region_0.append(key)
        if "leisure_count" in key:
            leisure_count.append(key)
        if "male" in key:
            male_female_all.append(key)
        if "female" in key:
            male_female_all.append(key)
        if "all" in key:
            male_female_all.append(key)
        if "prom_part" in key:
            if val>=0:
                prom_part.append(key)
            elif val<0:
                prom_part_0.append(key)
        if "green_part" in key:
            green_part.append(key)
        if "market_count" in key:
            market_count.append(key)
        if "mosque_count" in key:
            mosque_count.append(key)
    
    X['cafe_3'] = X[cafe_3].sum(axis=1)
    X_t['cafe_3'] = X_t[cafe_3].sum(axis=1)
    
    X['cafe_2_3'] = X[cafe_2_3].sum(axis=1)
    X_t['cafe_2_3'] = X_t[cafe_2_3].sum(axis=1)
    
    X['cafe_1_2'] = X[cafe_1_2].sum(axis=1)
    X_t['cafe_1_2'] = X_t[cafe_1_2].sum(axis=1)
    
    X['cafe_0_1'] = X[cafe_0_1].sum(axis=1)
    X_t['cafe_0_1'] = X_t[cafe_0_1].sum(axis=1)
    
    X['cafe_0'] = X[cafe_0].sum(axis=1)
    X_t['cafe_0'] = X_t[cafe_0].sum(axis=1)
    
    X['trc_count'] = X[trc_count].sum(axis=1)
    X_t['trc_count'] = X_t[trc_count].sum(axis=1)
    
    X['trc_sqm'] = X[trc_sqm].sum(axis=1)
    X_t['trc_sqm'] = X_t[trc_sqm].sum(axis=1)
    
    X['sport_count'] = X[sport_count].sum(axis=1)
    X_t['sport_count'] = X_t[sport_count].sum(axis=1)
    
    X['office_count'] = X[office_count].sum(axis=1)
    X_t['office_count'] = X_t[office_count].sum(axis=1)
    
    X['office_sqm'] = X[office_sqm].sum(axis=1)
    X_t['office_sqm'] = X_t[office_sqm].sum(axis=1)
    
    X['km_3'] = X[km_3].sum(axis=1)
    X_t['km_3'] = X_t[km_3].sum(axis=1)
    
    X['km_2_3'] = X[km_2_3].sum(axis=1)
    X_t['km_2_3'] = X_t[km_2_3].sum(axis=1)
    
    X['km_1_2'] = X[km_1_2].sum(axis=1)
    X_t['km_1_2'] = X_t[km_1_2].sum(axis=1)
    
    X['km_0_1'] = X[km_0_1].sum(axis=1)
    X_t['km_0_1'] = X_t[km_0_1].sum(axis=1)
    
    X['church_count'] = X[church_count].sum(axis=1)
    X_t['church_count'] = X_t[church_count].sum(axis=1)
    
    X['region_2'] = X[region_2].sum(axis=1)
    X_t['region_2'] = X_t[region_2].sum(axis=1)
    
    X['region_0_2'] = X[region_0_2].sum(axis=1)
    X_t['region_0_2'] = X_t[region_0_2].sum(axis=1)
    
    X['region_0'] = X[region_0].sum(axis=1)
    X_t['region_0'] = X_t[region_0].sum(axis=1)
    
    X['leisure_count'] = X[leisure_count].sum(axis=1)
    X_t['leisure_count'] = X_t[leisure_count].sum(axis=1)
    
    X['male_female_all'] = X[male_female_all].sum(axis=1)
    X_t['male_female_all'] = X_t[male_female_all].sum(axis=1)
    
    X['prom_part'] = X[prom_part].sum(axis=1)
    X_t['prom_part'] = X_t[prom_part].sum(axis=1)
    
    X['prom_part_0'] = X[prom_part_0].sum(axis=1)
    X_t['prom_part_0'] = X_t[prom_part_0].sum(axis=1)
    
    X['green_part'] = X[green_part].sum(axis=1)
    X_t['green_part'] = X_t[green_part].sum(axis=1)
    
    X['market_count'] = X[market_count].sum(axis=1)
    X_t['market_count'] = X_t[market_count].sum(axis=1)
    
    X['mosque_count'] = X[mosque_count].sum(axis=1)
    X_t['mosque_count'] = X_t[mosque_count].sum(axis=1)
    
    '''
    del_cols = cafe_3 + cafe_2_3 + cafe_1_2 + cafe_0_1 + cafe_0 + km_3 + km_2_3 + km_1_2 + km_0_1 + region_2 + region_0_2 + region_0 + trc_count + sport_count + office_count + trc_sqm + office_sqm + church_count + leisure_count + prom_part + prom_part_0 + green_part + male_female_all + market_count+mosque_count
    all_cols = X.columns
    for c in all_cols:
        if c in del_cols:
            X.drop(c, axis=1, inplace=True)
            X_t.drop(c, axis=1, inplace=True)
    print(X.shape)
    print(X_t.shape)'''
    
    X = pd.concat([X_ex, X], axis = 1)
    X_t = pd.concat([X_t_ex, X_t], axis = 1)
    #print(X.shape)
    #print(X_t.shape)
    
    '''
    corr_spear = X.iloc[:, 1:].corr(method='spearman')
    cor_dict = corr_spear['price_doc'].to_dict()
    del cor_dict['price_doc']
    print("List the numerical features decendingly by their correlation with Sale Price:\n")
    for ele in sorted(cor_dict.items(), key = lambda x: -abs(x[1])):
        print("{0}: \t{1}".format(*ele))
    '''
    
    print("add spearman features done.")
    
    return X, X_t

def main():
    train = pd.read_csv('data/train.csv', parse_dates=['timestamp'])
    test = pd.read_csv('data/test.csv', parse_dates=['timestamp'])
    
    train, test = clean_data(train, test)
    train, test = add_features(train, test)
    X, X_t = add_spearman_features(train, test)
    
    id_test = test.id
    from sklearn import preprocessing
    y_train = train["price_doc"]
    wts = 1 - .47*(y_train == 1e6)
    x_train = X.drop(["timestamp", "price_doc"], axis=1)
    x_test = X_t.drop(["timestamp"], axis=1)
    
    for c in x_train.columns:
        if x_train[c].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(x_train[c].values)) 
            x_train[c] = lbl.transform(list(x_train[c].values))
            #x_train.drop(c,axis=1,inplace=True)
            x_test[c] = lbl.transform(list(x_test[c].values))
            #x_test.drop(c,axis=1,inplace=True)  
    
    import xgboost as xgb
    xgb_params = {
        'eta': 0.05,
        'max_depth': 5,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1
    }
    
    dtrain = xgb.DMatrix(x_train, y_train, weight=wts)
    #dtrain = xgb.DMatrix(x_train, y_train)
    dtest = xgb.DMatrix(x_test)
    cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
        verbose_eval=50, show_stdv=False)
    #cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()
    
    num_boost_rounds = len(cv_output)#593
    model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)
    
    #fig, ax = plt.subplots(1, 1, figsize=(8, 13))
    #xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)
    
    y_predict = model.predict(dtest)
    jason_model_output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})
    jason_model_output.to_csv('jason_model_output.csv', index=False)
    
    train = pd.read_csv('data/train.csv', parse_dates=['timestamp'])
    test = pd.read_csv('data/test.csv', parse_dates=['timestamp'])
    
    add_decomposition_features(train, test)
    
    print(train.shape)
    print(test.shape)
'''
train = pd.read_csv('data/train.csv', parse_dates=['timestamp'])
test = pd.read_csv('data/test.csv', parse_dates=['timestamp'])

train, test = clean_data(train, test)
print(train.shape, test.shape)
train, test = add_decomposition_features(train, test)
print(train.shape, test.shape)'''
