# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 08:07:08 2017

@author: Administrator
"""

import gc
import numpy as np
import pandas as pd

lgb_sub_union_sub = pd.read_csv(r'submission/lgb/lgb_sub_union_sub.txt', sep=' ', header=None)
lgb_sub = pd.read_csv(r'submission/lgb/lgb_sub.txt',  sep=' ', header=None)

lgb_sub_union_all = lgb_sub_union_sub.copy()
lgb_sub_union_all[2] = lgb_sub_union_sub[2]*0.55 + lgb_sub[2]*0.45   #306370.0772
lgb_sub_union_all.to_csv(r'submission/combin/re_lgb_sub_union_all.txt', sep=' ', index=False, header=None)



























