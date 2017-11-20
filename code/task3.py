# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 16:16:13 2017

@author: Administrator
"""

import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid",{"font.sans-serif":['simhei', 'Arial']})
from collections import Counter

#train data 
p6M_mdl = pd.read_csv(r'data\lexin_train\p6M_mdl.csv', low_memory=False, nrows=None)
dep_mdl = pd.read_csv(r'data\lexin_train\dep_mdl.csv', low_memory=False, nrows=None)
dep_mdl = dep_mdl[['fuid_md5', 'dep', 'actual_od_brw_f6m']]

#test data 
p6M_offtime = pd.read_csv(r'data\lexin_test\p6M_offtime.csv', low_memory=False, nrows=None)
dep_offtime = pd.read_csv(r'submission\combin\lgb_sub.txt', low_memory=False, nrows=None, header=None, sep=' ')
dep_offtime.columns = dep_mdl.columns
dep_offtime['dep'] = dep_offtime['dep'].apply(lambda x: 1 if x > 0.5 else 0)

'''
前6月信用额度：credit_limit = C
前6月额度使用率：utilization = U
前6月逾期数：overdue_num = O
未来6月逾期率：dep = D
未来6月平均消费额：expend_mean  =   E
未来6月额度使用率：new_utilization = NU
 NU = E / C      
未来6月信用额度：new_credit_limit = NC
'''

train = pd.merge(p6M_mdl, dep_mdl, on='fuid_md5', how='left')
train['foverdue_payed_cyc'] = train['foverdue_payed_cyc'].apply(lambda x: 1 if x>0 else 0)

train_temp = train[['fuid_md5','credit_limit','futilization','dep']]\
                                              .groupby('fuid_md5')\
                                              .mean()\
                                              .reset_index()
train_features = train[['fuid_md5','foverdue_payed_cyc']]\
                                              .groupby('fuid_md5')\
                                              .sum()\
                                              .reset_index()\
                                              .rename(columns={'foverdue_payed_cyc':'overdue_num'})
train_temp = pd.merge(train_temp, train_features, on='fuid_md5', how='left')
train_temp['expend_mean'] = dep_mdl.actual_od_brw_f6m.values / 6
train_temp['new_utilization'] = train_temp['expend_mean'] / train_temp['credit_limit']  
# train_temp['futilization_mean'] = (train_temp['futilization']*5 + train_temp['new_utilization']) / 6
train_temp['futilization_mean'] = (train_temp['futilization']*6 + train_temp['new_utilization']) / 7
train_temp['new_credit_limit_A'] = train_temp['credit_limit'] * train_temp['futilization_mean']

train_temp['new_credit_limit_B'] = (train_temp['credit_limit'] * train_temp['futilization_mean'] + train_temp['credit_limit']) / 2

train_temp.to_csv(r'train_task3.csv', index=False)

'''优质用户：overdue_num=0、dep<0.5、(futilization>1、new_utilization>1= futilization_mean>1)'''
temp = train_temp[(train_temp['overdue_num'] == 0) & (train_temp['dep'] < 0.5) & (train_temp['futilization_mean'] > 1)]
g = sns.FacetGrid(data=temp, size=6)
g.map(sns.distplot, 'credit_limit', kde=False, color='b', bins=50)
plt.title('优质用户：额度调优前',fontsize=15)
plt.xlabel('credit_limit',fontsize=15)
plt.ylabel('count',fontsize=15)
plt.show()

plt.figure()
g = sns.FacetGrid(data=temp, size=6)
g.map(sns.distplot, 'new_credit_limit_A', kde=False, color='r', bins=50)
plt.title('优质用户：额度调优后',fontsize=15)
plt.xlabel('new_credit_limit',fontsize=15)
plt.ylabel('count',fontsize=15)
plt.show()

'''普通和流失用户：overdue_num=0、dep<0.5、(futilization>1、new_utilization>1= futilization_mean>1)'''
temp = train_temp[(train_temp['overdue_num'] == 0) & (train_temp['dep'] < 0.5) & (train_temp['futilization_mean'] < 1)]
g = sns.FacetGrid(data=temp, size=6)
g.map(sns.distplot, 'credit_limit', kde=False, color='b', bins=50)
plt.title('普通和流失用户：额度调优前',fontsize=15)
plt.xlabel('credit_limit',fontsize=15)
plt.ylabel('count',fontsize=15)
plt.show()

plt.figure()
g = sns.FacetGrid(data=temp, size=6)
g.map(sns.distplot, 'new_credit_limit_B', kde=False, color='r', bins=50)
plt.title('普通和流失用户：额度调优后',fontsize=15)
plt.xlabel('new_credit_limit',fontsize=15)
plt.ylabel('count',fontsize=15)
plt.show()

'''优质用户：overdue_num=0、dep<0.5、(futilization>1、new_utilization>1= futilization_mean>1)'''
temp = train_temp[(train_temp['overdue_num'] == 0) & (train_temp['dep'] < 0.5) & (train_temp['futilization_mean'] > 1)]
t1 = temp.shape[0]
tt1 = sum(temp.credit_limit)
ttt1 = sum(temp.new_credit_limit_A)

'''普通和流失用户：overdue_num=0、dep<0.5、(futilization<1、new_utilization<1= futilization_mean<1)'''
temp = train_temp[(train_temp['overdue_num'] == 0) & (train_temp['dep'] < 0.5) & (train_temp['futilization_mean'] < 1)]
t2 = temp.shape[0]
tt2 = sum(temp.credit_limit)
ttt2 = sum(temp.new_credit_limit_B)

'''不良用户1：overdue_num>5 & dep>0.5''' #直接赋值0
temp = train_temp[(train_temp['overdue_num'] > 3) & (train_temp['dep'] > 0.5)]
t3 = temp.shape[0]
tt3 = sum(temp.credit_limit)
ttt3 = 0

'''不良用户2：''' #保持原来 
t4 = 50000 - (t1 + t2 + t3)
tt4 = sum(train_temp.credit_limit) - (tt1 + tt2 + tt3)
ttt4 = tt4

print('train四类用户的用户数量：')
print([t1,t2,t3,t4,t1+t2+t3+t4])
print('train额度调优前分布：')
print([tt1,tt2,tt3,tt4,tt1+tt2+tt3+tt4])
print('train额度调优后分布：')
print([ttt1,ttt2,ttt3,ttt4,ttt1+ttt2+ttt3+ttt4])
print('train总额度：')
print(sum(train_temp.credit_limit))



#####################################################
#####################################################
#####################################################
print('===========================================================')
test = pd.merge(p6M_offtime, dep_offtime, on='fuid_md5', how='left')
test['foverdue_payed_cyc'] = test['foverdue_payed_cyc'].apply(lambda x: 1 if x>0 else 0)

test_temp = test[['fuid_md5','credit_limit','futilization','dep']]\
                                              .groupby('fuid_md5')\
                                              .mean()\
                                              .reset_index()
test_features = test[['fuid_md5','foverdue_payed_cyc']]\
                                              .groupby('fuid_md5')\
                                              .sum()\
                                              .reset_index()\
                                              .rename(columns={'foverdue_payed_cyc':'overdue_num'})
test_temp = pd.merge(test_temp, test_features, on='fuid_md5', how='left')
test_temp['expend_mean'] = dep_offtime.actual_od_brw_f6m.values / 6
test_temp['new_utilization'] = test_temp['expend_mean'] / test_temp['credit_limit']  
# test_temp['futilization_mean'] = (test_temp['futilization']*5 + test_temp['new_utilization']) / 6
test_temp['futilization_mean'] = (test_temp['futilization']*6 + test_temp['new_utilization']) / 7
         
test_temp['new_credit_limit_A'] = test_temp['credit_limit'] * test_temp['futilization_mean']

test_temp['new_credit_limit_B'] = (test_temp['credit_limit'] * test_temp['futilization_mean'] + test_temp['credit_limit']) / 2

test_temp.to_csv(r'test_task3.csv', index=False)

'''优质用户：overdue_num=0、dep<0.5、(futilization>1、new_utilization>1= futilization_mean>1)'''
temp = test_temp[(test_temp['overdue_num'] == 0) & (test_temp['dep'] < 0.5) & (test_temp['futilization_mean'] > 1)]
g = sns.FacetGrid(data=temp, size=6)
g.map(sns.distplot, 'credit_limit', kde=False, color='b', bins=50)
plt.title('优质用户：额度调优前',fontsize=15)
plt.xlabel('credit_limit',fontsize=15)
plt.ylabel('count',fontsize=15)
plt.show()

plt.figure()
g = sns.FacetGrid(data=temp, size=6)
g.map(sns.distplot, 'new_credit_limit_A', kde=False, color='r', bins=50)
plt.title('优质用户：额度调优后',fontsize=15)
plt.xlabel('new_credit_limit',fontsize=15)
plt.ylabel('count',fontsize=15)
plt.show()


'''普通和流失用户：overdue_num=0、dep<0.5、(futilization>1、new_utilization>1= futilization_mean>1)'''
temp = test_temp[(test_temp['overdue_num'] == 0) & (test_temp['dep'] < 0.5) & (test_temp['futilization_mean'] < 1)]
g = sns.FacetGrid(data=temp, size=6)
g.map(sns.distplot, 'credit_limit', kde=False, color='b', bins=50)
plt.title('普通和流失用户：额度调优前',fontsize=15)
plt.xlabel('credit_limit',fontsize=15)
plt.ylabel('count',fontsize=15)
plt.show()

plt.figure()
g = sns.FacetGrid(data=temp, size=6)
g.map(sns.distplot, 'new_credit_limit_B', kde=False, color='r', bins=50)
plt.title('普通和流失用户：额度调优后',fontsize=15)
plt.xlabel('new_credit_limit',fontsize=15)
plt.ylabel('count',fontsize=15)
plt.show()


'''优质用户：overdue_num=0、dep<0.5、(futilization>1、new_utilization>1= futilization_mean>1)'''
temp = test_temp[(test_temp['overdue_num'] == 0) & (test_temp['dep'] < 0.5) & (test_temp['futilization_mean'] > 1)]
t1 = temp.shape[0]
tt1 = sum(temp.credit_limit)
ttt1 = sum(temp.new_credit_limit_A)

'''普通和流失用户：overdue_num=0、dep<0.5、(futilization<1、new_utilization<1= futilization_mean<1)'''
temp = test_temp[(test_temp['overdue_num'] == 0) & (test_temp['dep'] < 0.5) & (test_temp['futilization_mean'] < 1)]
t2 = temp.shape[0]
tt2 = sum(temp.credit_limit)
ttt2 = sum(temp.new_credit_limit_B)

'''不良用户1：overdue_num>5 & dep>0.5''' #直接赋值0
temp = test_temp[(test_temp['overdue_num'] > 3) & (test_temp['dep'] > 0.5)]
t3 = temp.shape[0]
tt3 = sum(temp.credit_limit)
ttt3 = 0

'''不良用户2：''' #保持原来 
t4 = 50000 - (t1 + t2 + t3)
tt4 = sum(test_temp.credit_limit) - (tt1 + tt2 + tt3)
ttt4 = tt4

print('test四类用户的用户数量：')
print([t1,t2,t3,t4,t1+t2+t3+t4])
print('test额度调优前分布：')
print([tt1,tt2,tt3,tt4,tt1+tt2+tt3+tt4])
print('test额度调优后分布：')
print([ttt1,ttt2,ttt3,ttt4,ttt1+ttt2+ttt3+ttt4])
print('test总额度：')
print(sum(test_temp.credit_limit))
















