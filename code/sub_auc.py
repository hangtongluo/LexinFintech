# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 11:05:57 2017

@author: Administrator
"""

import gc
import numpy as np
import pandas as pd

lgb_sub_cl = pd.read_csv(r'submission/lgb/lgb_sub_cl.txt', nrows=None)

lgb_sub = lgb_sub_cl.copy()
lgb_sub['mae_pre'] = 0
lgb_sub = lgb_sub[['fuid_md5','auc_pre','mae_pre']]       
lgb_sub.to_csv(r'submission/lgb/lgb_sub.txt', sep=' ', index=False, header=None) #(最好的结果)




















