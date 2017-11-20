# -- coding: utf-8 --
import pandas as pd
import numpy as np
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
import matplotlib.pylab as plt
from imblearn.datasets import make_imbalance
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.grid_search import GridSearchCV

ud_train = pd.read_csv(r'features/ud_train.csv',encoding='gb2312')
ud_test = pd.read_csv(r'features/ud_test.csv',encoding='gb2312')
login_scene_mdl = pd.read_csv(r'features/login_scene_mdl.csv')
login_scene_offtime = pd.read_csv(r'features/login_scene_offtime.csv')
p6M_mdl = pd.read_csv(r'features/p6M_mdl_sta.csv')
p6M_offtime = pd.read_csv(r'features/p6M_offtime_sta.csv')
##########################################################################
train_od_in6m = pd.read_csv(r'features/train_od_in6m.csv', nrows=None)
test_od_in6m = pd.read_csv(r'features/test_od_in6m.csv', nrows=None)
train_p12M_statistics = pd.read_csv(r'features/train_p12M_statistics.csv', nrows=None)
test_p12M_statistics = pd.read_csv(r'features/test_p12M_statistics.csv', nrows=None)
#############################################################################

train = pd.merge(ud_train,login_scene_mdl,how='left',on='fuid_md5')
train = pd.merge(train,p6M_mdl,how='left',on='fuid_md5')
train = pd.merge(train,train_od_in6m,how='left',on='fuid_md5')
train = pd.merge(train,train_p12M_statistics,how='left',on='fuid_md5')

test = pd.merge(ud_test,login_scene_offtime,how='left',on='fuid_md5')
test = pd.merge(test,p6M_offtime,how='left',on='fuid_md5')
test = pd.merge(test,test_od_in6m,how='left',on='fuid_md5')
test = pd.merge(test,test_p12M_statistics,how='left',on='fuid_md5')

label = pd.read_csv(r'label.csv')
label=label.drop('Unnamed: 0',axis=1)

train = train.drop('dep',axis=1)
test = test.drop('dep',axis=1)
train = pd.merge(train,label,how='left',on='fuid_md5')
train = train.drop('counts',axis=1)
train = train.fillna(0)
test = test.fillna(0)

from sklearn.utils import shuffle 
train_zero=train[train.dep==0]
train_one=train[train.dep==1] 

train_zero=train_zero.sample(frac=0.33,random_state=20)      
train=pd.concat([train_zero,train_one],axis=0).reset_index()
train = shuffle(train)     

train = train.drop('index',axis=1)
y=train['dep']
x= train.drop(['fuid_md5','dep'],axis = 1)

xgb_train_cv = xgb.DMatrix(x, label=y)
params={
    'booster':'gbtree',
    'objective': 'binary:logistic',
   #'early_stopping_rounds':50,
    'scale_pos_weight': 1,
    #'n_estimators': 2000,
    'eval_metric': 'auc',
    'gamma':0,
    'max_depth':4,
   # 'lambda':50,
    'subsample':0.8,
    'colsample_bytree':0.8,
    'min_child_weight':1,
    'eta': 0.01,
    'nthread':-1, #32
    'alpha':0.1,
    'random_state':1000
}
#cvresult =xgb.cv(plst, xgb_train_cv,num_boost_round=2000,early_stopping_rounds=100, nfold=5, maximize=False, verbose_eval=True)

print ("跑到这里了xgb.train")
model = xgb.train(params, xgb_train_cv,num_boost_round=2000)
print ("跑到这里了save_model")
model.save_model('model/xgb.model') # 用于存储训练出的模型
print ("best best_ntree_limit",model.best_ntree_limit)   #did not save the best,why?
print ("best best_iteration",model.best_iteration) #get it?

feature_score = model.get_fscore()
feature_score = sorted(feature_score.items(), key=lambda x:x[1],reverse=True)
fs = []
for (key,value) in feature_score:
    fs.append("{0},{1}\n".format(key,value))
    
with open('feature_score.csv','w') as f:
        f.writelines("feature,score\n")
        f.writelines(fs)
		
result = pd.DataFrame(test['fuid_md5'])
test = test.drop('fuid_md5',axis = 1)
test = xgb.DMatrix(test)
sub = model.predict(test)

result['pre_auc'] = sub
result['pre_mae'] = 0

result.to_csv(r'submission/xgb/xgb_sub.txt', sep=' ', index=False, header=None) #(最好的结果)

	

#原始模型
################################################################################
#原始特征
#0.927346

#原始特征 +train_od_in6m
#0.928612

#原始特征 +train_od_in6m + train_p12M
#0.928354

#原始特征 +train_od_in6m + train_p12M_statistics
#0.9294










