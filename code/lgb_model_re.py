# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 10:05:24 2017

@author: Administrator
"""

import gc
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
import xgboost as xgb
import lightgbm as lgb
from utils import ont_hotTool

train_data = pd.read_csv(r'features/ud_train.csv',encoding='gb2312', nrows=None)
dep_mdl = pd.read_csv(r'data/lexin_train/dep_mdl.csv')
train_p12M = pd.read_csv(r'features/train_p12M.csv', nrows=None)
train_p6M_last_1_mean = pd.read_csv(r'features/train_p6M_last_1_mean.csv', nrows=None)
train_p6M_last_3_mean = pd.read_csv(r'features/train_p6M_last_3_mean.csv', nrows=None)
train_p6M_last_6_mean = pd.read_csv(r'features/train_p6M_last_6_mean.csv', nrows=None)
train_od_in6m = pd.read_csv(r'features/train_od_in6m.csv', nrows=None)
    
train_data = pd.merge(train_data, dep_mdl, on=['fuid_md5', 'dep'], how='left')
train_data = pd.merge(train_data, train_p12M, on='fuid_md5', how='left')
train_data = pd.merge(train_data, train_p6M_last_1_mean, on='fuid_md5', how='left')
train_data = pd.merge(train_data, train_p6M_last_3_mean, on='fuid_md5', how='left')
train_data = pd.merge(train_data, train_p6M_last_6_mean, on='fuid_md5', how='left')
train_data = pd.merge(train_data, train_od_in6m, on='fuid_md5', how='left')
train_data = train_data.drop(['p12M_one','p12M_two','p12M_three','p12M_four','p12M_five','p12M_six'],axis=1)

test_data = pd.read_csv(r'features/ud_test.csv',encoding='gb2312', nrows=None)
test_data = test_data.drop('dep', axis=1)
test_p12M = pd.read_csv(r'features/test_p12M.csv', nrows=None)
test_p6M_last_1_mean = pd.read_csv(r'features/test_p6M_last_1_mean.csv', nrows=None)
test_p6M_last_3_mean = pd.read_csv(r'features/test_p6M_last_3_mean.csv', nrows=None)
test_p6M_last_6_mean = pd.read_csv(r'features/test_p6M_last_6_mean.csv', nrows=None)
test_od_in6m = pd.read_csv(r'features/test_od_in6m.csv', nrows=None)

test_data = pd.merge(test_data, test_p12M, on='fuid_md5', how='left')
test_data = pd.merge(test_data, test_p6M_last_1_mean, on='fuid_md5', how='left')
test_data = pd.merge(test_data, test_p6M_last_3_mean, on='fuid_md5', how='left')
test_data = pd.merge(test_data, test_p6M_last_6_mean, on='fuid_md5', how='left')
test_data = pd.merge(test_data, test_od_in6m, on='fuid_md5', how='left')
test_data = test_data.drop(['p12M_one','p12M_two','p12M_three','p12M_four','p12M_five','p12M_six'],axis=1)

del train_p12M, train_p6M_last_1_mean, train_p6M_last_3_mean, train_p6M_last_6_mean
del test_p12M, test_p6M_last_1_mean, test_p6M_last_3_mean, test_p6M_last_6_mean
del train_od_in6m, test_od_in6m
gc.collect()

drop_colnames = ['fuid_md5']
target_colnames = ['dep', 'actual_od_brw_f6m', 'actual_od_brw_1stm',\
                  'actual_od_brw_2stm', 'actual_od_brw_3stm', 'actual_od_brw_4stm',\
                  'actual_od_brw_5stm', 'actual_od_brw_6stm']

#######################################################################################################
col = ['fuid_md5','dep','actual_od_brw_f6m','actual_od_brw_1stm','actual_od_brw_2stm','actual_od_brw_3stm','actual_od_brw_4stm', \
       'actual_od_brw_5stm','actual_od_brw_6stm','p12M_seven','p12M_eight','p12M_nine','p12M_ten','p12M_eleven','p12M_twelve']
#train_data[col].to_csv('look_data.csv')
look_train_data = train_data[col]
col = ['fuid_md5','p12M_seven','p12M_eight','p12M_nine','p12M_ten','p12M_eleven','p12M_twelve']
look_test_data = test_data[col]
#######################################################################################################

train_target = train_data[target_colnames]
train_data = train_data.drop(drop_colnames + target_colnames, axis=1)
lgb_sub_re = pd.DataFrame(test_data['fuid_md5'])
test_data = test_data.drop(drop_colnames, axis=1)

X_train, X_val, y_train, y_val = train_test_split(
                train_data, train_target['actual_od_brw_f6m'], test_size=0.2, random_state=2017)

lgb_train = lgb.Dataset(X_train, y_train)
lgb_val = lgb.Dataset(X_val, y_val)
test = test_data

params = {'boosting_type': 'gbdt', #'gbdt'
        'objective': 'regression',
        'metric': {'mae'}, #'l2', 
        'max_depth':6,
        'num_leaves': 80,
        'learning_rate': 0.01, #0.05
        'subsample':0.7, 
        'feature_fraction': 0.7,
        'bagging_fraction': 0.8,
        'bagging_freq': 10,
        'min_data_in_leaf':100,
        'num_threads':-1,
        'seed':2017
}

#模型训练
model = lgb.train(params,lgb_train,num_boost_round=2000,valid_sets=lgb_val,early_stopping_rounds=50)
model.save_model('model/lgbregression.txt')  #save model
    
#特征重要性                
#plt.figure()
#lgb.plot_importance(model, max_num_features=50)                            
#plt.savefig('lgbclassifier_importance.jpg')                

feature_importances = pd.Series(model.feature_importance(), model.feature_name()).sort_values(ascending=True)
pd.DataFrame(feature_importances).to_csv('model/lgb_re_feature_importances.csv')
plt.figure(figsize=(16,10))
feature_importances.plot(kind='barh', title='Feature Importances')
plt.xlabel('Feature Importance Score')
plt.savefig('lgbregression_importance.jpg')
#plt.show()          

#模型预测
pre = model.predict(test, num_iteration=model.best_iteration)
col = ['p12M_seven', 'p12M_eight','p12M_nine','p12M_ten','p12M_eleven','p12M_twelve']
lgb_sub_re['mae_pre'] = pre
lgb_sub_re['sum'] = look_test_data[col].sum(axis=1) / 2
lgb_sub_re.to_csv(r'submission/lgb/lgb_sub_re.txt', index=False)


########################################结果记录############################################
#p6M_last_1 ：valid_0's l1: 268588
#p6M_last_3 ：valid_0's l1: 271777
#p6M_last_6 ：valid_0's l1: 276748
#p6M_last_3，p6M_last_1 ：valid_0's l1: 267908
#p6M_last_1，p6M_last_6 ：valid_0's l1: 268922
#p6M_last_3，p6M_last_6 ：valid_0's l1: 271843
#p6M_last_6，p6M_last_3，p6M_last_1 ：valid_0's l1: 268275
#p6M_last_6，p6M_last_3，p6M_last_1，login_scene_last_1_mean valid_0's l1: 268188
#p6M_last_6，p6M_last_3，p6M_last_1，login_scene_last_2_mean valid_0's l1: 268188
#p6M_last_6，p6M_last_3，p6M_last_1，login_scene_last_3_mean valid_0's l1: 268188
#p6M_last_6，p6M_last_3，p6M_last_1，login_scene_last_136_mean valid_0's l1: 268024
#p6M_last_6，p6M_last_3，p6M_last_1，od_in6m valid_0's l1: 267220  ==3==
#p6M_last_6，p6M_last_3，p6M_last_1，p12M_statistics valid_0's l1: 268379


###################################lr=0.1##################################
#Early stopping, best iteration is:
#[19]    valid_0's l1: 267524


#Early stopping, best iteration is:(新的train_data) （311145.7969）
#[18]    valid_0's l1: 267336

#test 0.3
#Early stopping, best iteration is: (311049.0064)
#[20]    valid_0's l1: 264129
 

###################################lr=0.05##################################
#test:0.2
#Early stopping, best iteration is: (310109.7563)
#[40]    valid_0's l1: 267146


#Early stopping, best iteration is: #309138.8836 (用最近的6个月进行预测)(学习率位0.1)
#[19]    valid_0's l1: 267522


#Early stopping, best iteration is: (用最近的6个月进行预测)(学习率位0.01)#（最好成绩）
#[205]   valid_0's l1: 266344

































