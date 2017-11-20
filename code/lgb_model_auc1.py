# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 11:32:24 2017

@author: Administrator
"""

import gc
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
from imblearn.under_sampling import RandomUnderSampler 
import xgboost as xgb
import lightgbm as lgb


ud_train = pd.read_csv(r'features/ud_train.csv',encoding='gb2312', nrows=None)
ud_test = pd.read_csv(r'features/ud_test.csv',encoding='gb2312', nrows=None)
login_scene_mdl = pd.read_csv(r'features/login_scene_mdl.csv')
login_scene_offtime = pd.read_csv(r'features/login_scene_offtime.csv')
p6M_mdl = pd.read_csv(r'features/p6M_mdl_sta.csv')
p6M_offtime = pd.read_csv(r'features/p6M_offtime_sta.csv')
##############################################################################
train_od_in6m = pd.read_csv(r'features/train_od_in6m.csv', nrows=None)
test_od_in6m = pd.read_csv(r'features/test_od_in6m.csv', nrows=None)
train_p12M_statistics = pd.read_csv(r'features/train_p12M_statistics.csv', nrows=None)
test_p12M_statistics = pd.read_csv(r'features/test_p12M_statistics.csv', nrows=None)
##############################################################################

train_data = pd.merge(ud_train,login_scene_mdl,how='left',on='fuid_md5')
train_data = pd.merge(train_data,p6M_mdl,how='left',on='fuid_md5')
train_data = pd.merge(train_data,train_od_in6m,how='left',on='fuid_md5')
train_data = pd.merge(train_data,train_p12M_statistics,how='left',on='fuid_md5')

test_data = pd.merge(ud_test,login_scene_offtime,how='left',on='fuid_md5')
test_data = pd.merge(test_data,p6M_offtime,how='left',on='fuid_md5')
test_data = pd.merge(test_data,test_od_in6m,how='left',on='fuid_md5')
test_data = pd.merge(test_data,test_p12M_statistics,how='left',on='fuid_md5')

del ud_train, login_scene_mdl, p6M_mdl, train_od_in6m, train_p12M_statistics
del ud_test, login_scene_offtime, p6M_offtime, test_od_in6m, test_p12M_statistics
gc.collect()

dropcolnames = ['fuid_md5']
targetcolnames = ['dep']

train_target = train_data[targetcolnames]
train_data = train_data.drop(dropcolnames + targetcolnames, axis=1)
lgb_sub_cl = pd.DataFrame(test_data['fuid_md5'])
test_data = test_data.drop(dropcolnames + targetcolnames, axis=1)
train_data = train_data.fillna(-999)
test_data = test_data.fillna(-999)

X_train, X_val, y_train, y_val = train_test_split(
                train_data, train_target['dep'], test_size=0.2, random_state=2017)


lgb_train = lgb.Dataset(X_train, y_train)
lgb_val = lgb.Dataset(X_val, y_val)
test = test_data

del train_data, test_data
del X_train, X_val, y_train, y_val
gc.collect()


params={'boosting_type':'gbdt',
	    'objective': 'binary',
	    'metric':'auc',
	    'max_depth':6, #6,8
	    'num_leaves':2**6, #80, 
	    'lambda_l2':1,
	    'subsample':0.7,
	    'learning_rate': 0.1, #0.01 (0.1)
	    'feature_fraction':0.7,
	    'bagging_fraction':0.8,
	    'bagging_freq':10,
	    'num_threads':-1,
        'seed':2017,
#        'min_data_in_leaf': 100
}

#模型训练
model = lgb.train(params,lgb_train,num_boost_round=2000,valid_sets=lgb_val,early_stopping_rounds=50)
model.save_model('model/lgbclassifier.txt')  #save model             

feature_importances = pd.Series(model.feature_importance(), model.feature_name()).sort_values(ascending=True)
pd.DataFrame(feature_importances).to_csv('model/lgb_cl_feature_importances.csv')
plt.figure(figsize=(16,10))
feature_importances.plot(kind='barh', title='Feature Importances')
plt.xlabel('Feature Importance Score')
plt.savefig('lgbclassifier_importance.jpg')
#plt.show()  

#模型预测                
pre = model.predict(test, num_iteration=model.best_iteration)
lgb_sub_cl['auc_pre'] = pre
lgb_sub_cl.to_csv(r'submission/lgb/lgb_sub_cl.txt', index=False) #(最好的结果)



#没有采样数据
#################################test_size=0.2############################################
#Early stopping, best iteration is: 0.922718
#[115]   valid_0's auc: 0.933411

#Early stopping, best iteration is:  +train_od_in6m 
#[114]   valid_0's auc: 0.933842
 
#Early stopping, best iteration is: +train_od_in6m +train_p12M_statistics 
#[93]    valid_0's auc: 0.934563

#Early stopping, best iteration is:  +train_od_in6m +train_p12M
#[119]   valid_0's auc: 0.935988 







#################################test_size=0.1############################################
#Early stopping, best iteration is:  0.924646
#[83]    valid_0's auc: 0.934709

#Early stopping, best iteration is: +train_od_in6m 0.926514  ========2========
#[72]    valid_0's auc: 0.935669 

#Early stopping, best iteration is: +train_od_in6m +train_p12M (估计0.9285)===0.925014========
#[88]    valid_0's auc: 0.937402 

#Early stopping, best iteration is: +train_od_in6m +train_p12M +train_p12M_statistics (估计0.9280)
#[84]    valid_0's auc: 0.937131

#Early stopping, best iteration is: +train_od_in6m +train_p12M +train_login_scene_last_3_mean (估计0.9260)
#[79]    valid_0's auc: 0.936064 

#Early stopping, best iteration is:#0.923815 (增加转化率和经纬度)
#[90]    valid_0's auc: 0.935383


#Early stopping, best iteration is: +train_od_in6m +train_p12M_statistics  0.927437 ========1========
#[73]    valid_0's auc: 0.934935






#######################################0.1，lr=0.01######################################################
#Early stopping, best iteration is:
#[730]   valid_0's auc: 0.936827


#Early stopping, best iteration is: +train_od_in6m +train_p12M (估计0.9290)
#[731]   valid_0's auc: 0.938227

 

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 



