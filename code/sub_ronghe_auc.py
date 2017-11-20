# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 23:27:30 2017

@author: Administrator
"""

import gc
import numpy as np
import pandas as pd

xgb_sub = pd.read_csv(r'submission/xgb/xgb_sub.txt', nrows=None, header=None, sep=' ')
#lgb_sub_cl1 = pd.read_csv(r'lgb_sub_cl.txt', nrows=None, header=None, sep=' ')
##################################################################################
sub_2 = pd.read_csv(r'submission/combin/sub_2.txt', nrows=None, header=None, sep='\t')
submission_v5_10 = pd.read_csv(r'submission/combin/submission_v5_10.txt', nrows=None, header=None, sep='\t')
submission_v3_4 = pd.read_csv(r'submission/combin/submission_v3_4.txt', nrows=None, header=None, sep='\t')
##############################################################################################
lgb_sub_cl = pd.read_csv(r'submission/lgb/lgb_sub.txt', sep=' ', header=None)

sub = xgb_sub.copy()
sub[1] = xgb_sub[1]*0.7 + sub_2[1]*0.3
sub.to_csv(r'submission/combin/ronghe_sub.txt', sep=' ', index=False, header=None) #0.930504

sub = xgb_sub.copy()
sub[1] = xgb_sub[1]*0.7 + submission_v5_10[1]*0.3
sub.to_csv(r'submission/combin/ronghe_sub1.txt', sep=' ', index=False, header=None) #0.93052   

sub = xgb_sub.copy()
sub[1] = xgb_sub[1]*0.65 + submission_v5_10[1]*0.35
sub.to_csv(r'submission/combin/ronghe_sub1_0.65_0.35.txt', sep=' ', index=False, header=None) #0.930644

sub = xgb_sub.copy()
sub[1] = xgb_sub[1]*0.55 + lgb_sub_cl[1]*0.45
sub.to_csv(r'submission/combin/ronghe_sub1_lgb_sub_cl.txt', sep=' ', index=False, header=None) #0.930751

ronghe_sub1_lgb_sub_cl = pd.read_csv(r'submission/combin/ronghe_sub1_lgb_sub_cl.txt', nrows=None, header=None, sep=' ')
sub = ronghe_sub1_lgb_sub_cl.copy()
sub[1] = ronghe_sub1_lgb_sub_cl[1]*0.7 + submission_v5_10[1]*0.3
sub.to_csv(r'submission/combin/ronghe_sub_all.txt', sep=' ', index=False, header=None) #0.931658

ronghe_sub_all = pd.read_csv(r'submission/combin/ronghe_sub_all.txt', nrows=None, header=None, sep=' ')
sub = ronghe_sub_all.copy()
sub[1] = ronghe_sub_all[1]*1.2 + submission_v3_4[1]*(-0.2)
sub.to_csv(r'submission/combin/ronghe_sub_all_2.txt', sep=' ', index=False, header=None) #0.932403

#sub = ronghe_sub_all.copy()
#sub[1] = ronghe_sub_all[1]*1.5 + submission_v3_4[1]*(-0.3) + lgb_sub_cl1[1]*(-0.2)
#sub.to_csv(r'submission/combin/ronghe_sub_all_3.txt', sep=' ', index=False, header=None) 




























