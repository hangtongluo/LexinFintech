# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 10:45:41 2017

@author: Administrator
"""

import gc
import numpy as np
import pandas as pd

lgb_sub_cl = pd.read_csv(r'submission/lgb/lgb_sub_cl.txt', nrows=None)
lgb_sub_re = pd.read_csv(r'submission/lgb/lgb_sub_re.txt', nrows=None)
union_user = pd.read_csv(r'submission/lgb/union_user.csv', nrows=None) 
union_user.columns = ['fuid_md5', 'auc_pre', 'mae_pre']
#union_user['auc_pre'] = union_user['auc_pre'].apply(lambda x: lgb_sub_cl['auc_pre'].min() if x==0 else lgb_sub_cl['auc_pre'].max())
#################################################################################################
#lgb_sub_re['sum'] = lgb_sub_re['sum'].apply(lambda x: 1 if x > 0 else 0)
#lgb_sub_re['mae_pre'] = lgb_sub_re['mae_pre'] * lgb_sub_re['sum']
#lgb_sub_re = lgb_sub_re.drop('sum', axis=1)
#################################################################################################

lgb_sub = pd.merge(lgb_sub_cl, lgb_sub_re, on='fuid_md5', how='left')
lgb_sub.to_csv(r'submission/lgb/lgb_sub.txt', sep=' ', index=False, header=None)

user_name = list(set(lgb_sub['fuid_md5']) - set(union_user['fuid_md5']))
lgb_sub_union = pd.DataFrame({'fuid_md5':user_name})
lgb_sub_union = pd.merge(lgb_sub_union, lgb_sub, on='fuid_md5', how='left')
lgb_sub_union = pd.concat([lgb_sub_union, union_user], axis=0)
lgb_sub_union_sub = pd.merge(pd.DataFrame(lgb_sub['fuid_md5']), lgb_sub_union, on='fuid_md5', how='left')
lgb_sub_union_sub.to_csv(r'submission/lgb/lgb_sub_union_sub.txt', sep=' ', index=False, header=None)



#lgb_sub_union_sub : mae为306738.1128 (lr=0.1)

#lgb_sub_union_sub : mae为305596.3226 (lr=0.01)



















