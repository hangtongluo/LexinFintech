# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 22:53:06 2017

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

train_data = pd.read_csv(r'features/ud_train.csv',encoding='gb2312', nrows=None)
train_p12M = pd.read_csv(r'features/train_p12M.csv', nrows=None)
train_p6M_last_1_mean = pd.read_csv(r'features/train_p6M_last_1_mean.csv', nrows=None)
train_p6M_last_3_mean = pd.read_csv(r'features/train_p6M_last_3_mean.csv', nrows=None)
train_p6M_last_6_mean = pd.read_csv(r'features/train_p6M_last_6_mean.csv', nrows=None)
train_od_in6m = pd.read_csv(r'features/train_od_in6m.csv', nrows=None)
#train_p6M_features_last_1 = pd.read_csv(r'features/train_p6M_features_last_1.csv', nrows=None)
#train_p6M_features_last_3 = pd.read_csv(r'features/train_p6M_features_last_3.csv', nrows=None)
#train_p6M_features_last_6 = pd.read_csv(r'features/train_p6M_features_last_6.csv', nrows=None)
    
train_data = pd.merge(train_data, train_p12M, on='fuid_md5', how='left')
train_data = pd.merge(train_data, train_p6M_last_1_mean, on='fuid_md5', how='left')
train_data = pd.merge(train_data, train_p6M_last_3_mean, on='fuid_md5', how='left')
train_data = pd.merge(train_data, train_p6M_last_6_mean, on='fuid_md5', how='left')
train_data = pd.merge(train_data, train_od_in6m, on='fuid_md5', how='left')
#train_data = pd.merge(train_data, train_p6M_features_last_1, on='fuid_md5', how='left')
#train_data = pd.merge(train_data, train_p6M_features_last_3, on='fuid_md5', how='left')
#train_data = pd.merge(train_data, train_p6M_features_last_6, on='fuid_md5', how='left')

test_data = pd.read_csv(r'features/ud_test.csv',encoding='gb2312', nrows=None)
test_p12M = pd.read_csv(r'features/test_p12M.csv', nrows=None)
test_p6M_last_1_mean = pd.read_csv(r'features/test_p6M_last_1_mean.csv', nrows=None)
test_p6M_last_3_mean = pd.read_csv(r'features/test_p6M_last_3_mean.csv', nrows=None)
test_p6M_last_6_mean = pd.read_csv(r'features/test_p6M_last_6_mean.csv', nrows=None)
test_od_in6m = pd.read_csv(r'features/test_od_in6m.csv', nrows=None)
#test_p6M_features_last_1 = pd.read_csv(r'features/test_p6M_features_last_1.csv', nrows=None)
#test_p6M_features_last_3 = pd.read_csv(r'features/test_p6M_features_last_3.csv', nrows=None)
#test_p6M_features_last_6 = pd.read_csv(r'features/test_p6M_features_last_6.csv', nrows=None)

test_data = pd.merge(test_data, test_p12M, on='fuid_md5', how='left')
test_data = pd.merge(test_data, test_p6M_last_1_mean, on='fuid_md5', how='left')
test_data = pd.merge(test_data, test_p6M_last_3_mean, on='fuid_md5', how='left')
test_data = pd.merge(test_data, test_p6M_last_6_mean, on='fuid_md5', how='left')
test_data = pd.merge(test_data, test_od_in6m, on='fuid_md5', how='left')
#test_data = pd.merge(test_data, test_p6M_features_last_1, on='fuid_md5', how='left')
#test_data = pd.merge(test_data, test_p6M_features_last_3, on='fuid_md5', how='left')
#test_data = pd.merge(test_data, test_p6M_features_last_6, on='fuid_md5', how='left')

#del train_p12M, train_p6M_last_1_mean, train_p6M_last_3_mean, train_p6M_last_6_mean
#del train_od_in6m, train_p6M_features_last_1, train_p6M_features_last_1, train_p6M_features_last_1
#del test_p12M, test_p6M_last_1_mean, test_p6M_last_3_mean, test_p6M_last_6_mean
#del test_od_in6m, test_p6M_features_last_1, test_p6M_features_last_3, test_p6M_features_last_6
#gc.collect()

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

########################################################################################
print('not Resampled dataset shape {}'.format(Counter(y_train)))
rus = RandomUnderSampler(random_state=2017, ratio=0.05)
X_res, y_res = rus.fit_sample(X_train, y_train)
X_train = pd.DataFrame(X_res, columns=X_train.columns)
y_train = y_res
print('Resampled dataset shape {}'.format(Counter(y_train)))
########################################################################################

lgb_train = lgb.Dataset(X_train, y_train)
lgb_val = lgb.Dataset(X_val, y_val)
test = test_data

params={'boosting_type':'gbdt',
	    'objective': 'binary',
	    'metric':'auc',
	    'max_depth':6,
	    'num_leaves':2**6, #80, 
	    'lambda_l2':1,
	    'subsample':0.7,
	    'learning_rate': 0.1, #0.1
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
  
#特征重要性
#plt.figure()
#lgb.plot_importance(model, max_num_features=50)                            
#plt.savefig('lgbclassifier_importance.jpg')               

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
lgb_sub_cl.to_csv(r'submission/lgb/lgb_sub_cl2.txt', index=False)# (最后融合使用的)


########################################结果记录############################################
#p6M_last_1 ：valid_0's auc: 0.928244
#p6M_last_3 ：valid_0's auc: 0.929776
#p6M_last_6 ：valid_0's auc: 0.924636
#p6M_last_3，p6M_last_1 ：valid_0's auc: 0.928895
#p6M_last_1，p6M_last_6 ：valid_0's auc: 0.928915
#p6M_last_3，p6M_last_6 ：valid_0's auc: 0.930591
#p6M_last_6，p6M_last_3，p6M_last_1 ：valid_0's auc: 0.931316
#p6M_last_6，p6M_last_3，p6M_last_1，login_scene_last_1_mean valid_0's auc: 0.931933
#p6M_last_6，p6M_last_3，p6M_last_1，login_scene_last_3_mean valid_0's auc: 0.930401
#p6M_last_6，p6M_last_3，p6M_last_1，login_scene_last_6_mean valid_0's auc: 0.93226
#p6M_last_6，p6M_last_3，p6M_last_1，od_in6m valid_0's auc: 0.933802  ==3==
#p6M_last_6，p6M_last_3，p6M_last_1，p12M_statistics valid_0's auc: 0.930475



########################################采样############################################
#Early stopping, best iteration is: （0.05）
#[167]   valid_0's auc: 0.934472






































