# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 14:18:55 2017

@author: Administrator
"""

import gc
import numpy as np
import pandas as pd

lgb_sub_cl = pd.read_csv(r'submission/lgb/lgb_sub_cl.txt', nrows=None)
lgb_sub_re = pd.read_csv(r'submission/lgb/lgb_sub_re.txt', nrows=None)


#################################################################################################
lgb_sub_re['sum'] = lgb_sub_re['sum'].apply(lambda x: 1 if x > 0 else 0)
lgb_sub_re['mae_pre'] = lgb_sub_re['mae_pre'] * lgb_sub_re['sum']
lgb_sub_re = lgb_sub_re.drop('sum', axis=1)
#################################################################################################

lgb_sub = pd.merge(lgb_sub_cl, lgb_sub_re, on='fuid_md5', how='left')
lgb_sub.to_csv(r'submission/lgb/lgb_sub.txt', sep=' ', index=False, header=None)





'''
第一次提交：两个任务分别取最好成绩
lgb_sub：mae为357608.3125，auc为0.798096
xgb_sub：mae为355487.5312，auc为0.789692
线下验证：
lgb_sub：mae为328149，auc为0.842509
xgb_sub：mae为329146，auc为0.832235


第二次提交：
lgb_sub：mae为317002.6562，auc为0.912436
xgb_sub：

lgb_sub：mae为268588，auc为0.928244
xgb_sub：mae为269704，auc为0.927569


第三次提交：
lgb_sub：mae为313575.5312，auc为0.9156
xgb_sub：mae为318505.6875，auc为0.9151

lgb_sub：mae为267220，auc为0.933802
xgb_sub：mae为270470，auc为0.933661


'''



























