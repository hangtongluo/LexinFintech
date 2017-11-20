# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 21:14:58 2017

@author: Administrator
"""

import gc 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
from utils import groupby_unique_Tool, groupby_sum_Tool, groupby_mean_Tool, groupby_std_Tool, groupby_max_Tool, groupby_min_Tool
from utils import groupby_sum_pivot_Tool, groupby_mean_pivot_Tool, groupby_min_pivot_Tool, groupby_max_pivot_Tool, groupby_std_pivot_Tool


###############################用户基本信息特征提取#############################
def ud_info_ext():
    train_dep_target = pd.read_csv(r'pro_data/train_dep_target.csv', nrows=None)
    train_ud = pd.read_csv(r'pro_data/train_ud.csv', nrows=None)
    test_ud = pd.read_csv(r'pro_data/test_ud.csv', nrows=None)
    
    #注册时间，审批时间，和毕业时间进行拆分（注册和审批成功相隔时间）
    temp = pd.to_datetime(train_ud.fpocket_auth_time) - pd.to_datetime(train_ud.fregister_time) 
    train_ud['register_pocket_diff_days'] = temp.apply(lambda x: x.days)
    train_ud['register_pocket_diff_minutes'] = temp.apply(lambda x: x.seconds // 60)
    
    temp = pd.to_datetime(train_ud.fcal_graduation) - pd.to_datetime(train_ud.fregister_time) 
    train_ud['register_graduation_diff_days'] = temp.apply(lambda x: x.days)
    train_ud['register_graduation_diff_minutes'] = temp.apply(lambda x: x.seconds // 60)
    
    temp = pd.to_datetime(train_ud.fcal_graduation) - pd.to_datetime(train_ud.fpocket_auth_time) 
    train_ud['pocket_graduation_diff_days'] = temp.apply(lambda x: x.days)
    train_ud['pocket_graduation_diff_minutes'] = temp.apply(lambda x: x.seconds // 60)
    
    
    temp = pd.to_datetime(test_ud.fpocket_auth_time) - pd.to_datetime(test_ud.fregister_time) 
    test_ud['register_pocket_diff_days'] = temp.apply(lambda x: x.days)
    test_ud['register_pocket_diff_minutes'] = temp.apply(lambda x: x.seconds // 60)
    
    temp = pd.to_datetime(test_ud.fcal_graduation) - pd.to_datetime(test_ud.fregister_time) 
    test_ud['register_graduation_diff_days'] = temp.apply(lambda x: x.days)
    test_ud['register_graduation_diff_minutes'] = temp.apply(lambda x: x.seconds // 60)
    
    temp = pd.to_datetime(test_ud.fcal_graduation) - pd.to_datetime(test_ud.fpocket_auth_time) 
    test_ud['pocket_graduation_diff_days'] = temp.apply(lambda x: x.days)
    test_ud['pocket_graduation_diff_minutes'] = temp.apply(lambda x: x.seconds // 60)
    
    
    train_data = pd.merge(train_ud, train_dep_target, on='fuid_md5', how='left')
    test_data = test_ud
    
    del train_ud, test_ud
    gc.collect()
    
    
    train_data.to_csv(r'features/train_data.csv', index=False)
    test_data.to_csv(r'features/test_data.csv', index=False)
    
    print("ud_info_ext finishing...")
    return train_data, test_data

###############################过去12个月月度订单金额#############################
def p12M_info_ext():
    train_p12M = pd.read_csv(r'pro_data/train_p12M.csv', nrows=None)
    test_p12M = pd.read_csv(r'pro_data/test_p12M.csv', nrows=None)
    test_p12M.columns = train_p12M.columns
    
    #先进行基本的数据变换处理（还有时间上的操作需要进行后续处理）
    def index_p12M(df):
        df['p12M_index'] = ['p12M_one','p12M_two','p12M_three','p12M_four','p12M_five','p12M_six',\
                            'p12M_seven','p12M_eight','p12M_nine','p12M_ten','p12M_eleven','p12M_twelve']
        return df
    
    train_p12M = train_p12M.groupby('fuid_md5')\
                                   .apply(index_p12M)\
                                   .pivot(columns='p12M_index', values='od_brw_f12m', index='fuid_md5')\
                                   .reset_index()
    train_p12M = train_p12M[['fuid_md5','p12M_one','p12M_two','p12M_three','p12M_four','p12M_five','p12M_six',\
                            'p12M_seven','p12M_eight','p12M_nine','p12M_ten','p12M_eleven','p12M_twelve']]
    
    
    
    test_p12M = test_p12M.groupby('fuid_md5')\
                                   .apply(index_p12M)\
                                   .pivot(columns='p12M_index', values='od_brw_f12m', index='fuid_md5')\
                                   .reset_index()
    test_p12M = test_p12M[['fuid_md5','p12M_one','p12M_two','p12M_three','p12M_four','p12M_five','p12M_six',\
                            'p12M_seven','p12M_eight','p12M_nine','p12M_ten','p12M_eleven','p12M_twelve']]
    
    train_p12M.to_csv(r'features/train_p12M.csv', index=False)
    test_p12M.to_csv(r'features/test_p12M.csv', index=False)

    print("p12M_info_ext finishing...")
    return train_p12M, test_p12M

def p12M_statistics_ext():
    train_p12M = pd.read_csv(r'pro_data/train_p12M.csv', nrows=None)
    test_p12M = pd.read_csv(r'pro_data/test_p12M.csv', nrows=None)
    test_p12M.columns = train_p12M.columns
    
    train_p12M['p_month'] = pd.to_datetime(train_p12M['pyear_month']).apply(lambda x: x.month)
    map_dict = {1:1,2:1,3:1,
                4:2,5:2,6:2,
                7:3,8:3,9:3,
                10:4,11:4,12:4}
    train_p12M['p_quarter'] = train_p12M['p_month'].map(map_dict)
    train_p12M['p_halfyear'] = train_p12M['p_month'].apply(lambda x: 1 if x<7 else 2)
    
    test_p12M['p_month'] = pd.to_datetime(test_p12M['pyear_month']).apply(lambda x: x.month)
    map_dict = {1:1,2:1,3:1,
                4:2,5:2,6:2,
                7:3,8:3,9:3,
                10:4,11:4,12:4}
    test_p12M['p_quarter'] = test_p12M['p_month'].map(map_dict)
    test_p12M['p_halfyear'] = test_p12M['p_month'].apply(lambda x: 1 if x<7 else 2)
    
    
    train_features = groupby_sum_Tool(train_p12M, 'fuid_md5', 'od_brw_f12m')
    train_temp = groupby_mean_Tool(train_p12M, 'fuid_md5', 'od_brw_f12m')
    train_features = pd.merge(train_features, train_temp, on='fuid_md5', how='left')
    
    train_temp = groupby_std_Tool(train_p12M, 'fuid_md5', 'od_brw_f12m')
    train_features = pd.merge(train_features, train_temp, on='fuid_md5', how='left')
    
    train_temp = groupby_max_Tool(train_p12M, 'fuid_md5', 'od_brw_f12m')
    train_features = pd.merge(train_features, train_temp, on='fuid_md5', how='left')
    
    train_temp = groupby_min_Tool(train_p12M, 'fuid_md5', 'od_brw_f12m')
    train_features = pd.merge(train_features, train_temp, on='fuid_md5', how='left')
    
    train_temp = groupby_sum_pivot_Tool(train_p12M, ['fuid_md5','p_quarter'], 'od_brw_f12m')
    train_features = pd.merge(train_features, train_temp, on='fuid_md5', how='left')
    
    train_temp = groupby_mean_pivot_Tool(train_p12M, ['fuid_md5','p_quarter'], 'od_brw_f12m')
    train_features = pd.merge(train_features, train_temp, on='fuid_md5', how='left')
    
    train_temp = groupby_min_pivot_Tool(train_p12M, ['fuid_md5','p_quarter'], 'od_brw_f12m')
    train_features = pd.merge(train_features, train_temp, on='fuid_md5', how='left')
    
    train_temp = groupby_max_pivot_Tool(train_p12M, ['fuid_md5','p_quarter'], 'od_brw_f12m')
    train_features = pd.merge(train_features, train_temp, on='fuid_md5', how='left')
    
    train_temp = groupby_std_pivot_Tool(train_p12M, ['fuid_md5','p_quarter'], 'od_brw_f12m')
    train_features = pd.merge(train_features, train_temp, on='fuid_md5', how='left')
    
    train_temp = groupby_sum_pivot_Tool(train_p12M, ['fuid_md5','p_halfyear'], 'od_brw_f12m')
    train_features = pd.merge(train_features, train_temp, on='fuid_md5', how='left')
    
    train_temp = groupby_mean_pivot_Tool(train_p12M, ['fuid_md5','p_halfyear'], 'od_brw_f12m')
    train_features = pd.merge(train_features, train_temp, on='fuid_md5', how='left')
    
    train_temp = groupby_min_pivot_Tool(train_p12M, ['fuid_md5','p_halfyear'], 'od_brw_f12m')
    train_features = pd.merge(train_features, train_temp, on='fuid_md5', how='left')
    
    train_temp = groupby_max_pivot_Tool(train_p12M, ['fuid_md5','p_halfyear'], 'od_brw_f12m')
    train_features = pd.merge(train_features, train_temp, on='fuid_md5', how='left')
    
    train_temp = groupby_std_pivot_Tool(train_p12M, ['fuid_md5','p_halfyear'], 'od_brw_f12m')
    train_features = pd.merge(train_features, train_temp, on='fuid_md5', how='left')
    
    
    
    test_features = groupby_sum_Tool(test_p12M, 'fuid_md5', 'od_brw_f12m')
    test_temp = groupby_mean_Tool(test_p12M, 'fuid_md5', 'od_brw_f12m')
    test_features = pd.merge(test_features, test_temp, on='fuid_md5', how='left')
    
    test_temp = groupby_std_Tool(test_p12M, 'fuid_md5', 'od_brw_f12m')
    test_features = pd.merge(test_features, test_temp, on='fuid_md5', how='left')
    
    test_temp = groupby_max_Tool(test_p12M, 'fuid_md5', 'od_brw_f12m')
    test_features = pd.merge(test_features, test_temp, on='fuid_md5', how='left')
    
    test_temp = groupby_min_Tool(test_p12M, 'fuid_md5', 'od_brw_f12m')
    test_features = pd.merge(test_features, test_temp, on='fuid_md5', how='left')
    
    test_temp = groupby_sum_pivot_Tool(test_p12M, ['fuid_md5','p_quarter'], 'od_brw_f12m')
    test_features = pd.merge(test_features, test_temp, on='fuid_md5', how='left')
    
    test_temp = groupby_mean_pivot_Tool(test_p12M, ['fuid_md5','p_quarter'], 'od_brw_f12m')
    test_features = pd.merge(test_features, test_temp, on='fuid_md5', how='left')
    
    test_temp = groupby_min_pivot_Tool(test_p12M, ['fuid_md5','p_quarter'], 'od_brw_f12m')
    test_features = pd.merge(test_features, test_temp, on='fuid_md5', how='left')
    
    test_temp = groupby_max_pivot_Tool(test_p12M, ['fuid_md5','p_quarter'], 'od_brw_f12m')
    test_features = pd.merge(test_features, test_temp, on='fuid_md5', how='left')
    
    test_temp = groupby_std_pivot_Tool(test_p12M, ['fuid_md5','p_quarter'], 'od_brw_f12m')
    test_features = pd.merge(test_features, test_temp, on='fuid_md5', how='left')
    
    test_temp = groupby_sum_pivot_Tool(test_p12M, ['fuid_md5','p_halfyear'], 'od_brw_f12m')
    test_features = pd.merge(test_features, test_temp, on='fuid_md5', how='left')
    
    test_temp = groupby_mean_pivot_Tool(test_p12M, ['fuid_md5','p_halfyear'], 'od_brw_f12m')
    test_features = pd.merge(test_features, test_temp, on='fuid_md5', how='left')
    
    test_temp = groupby_min_pivot_Tool(test_p12M, ['fuid_md5','p_halfyear'], 'od_brw_f12m')
    test_features = pd.merge(test_features, test_temp, on='fuid_md5', how='left')
    
    test_temp = groupby_max_pivot_Tool(test_p12M, ['fuid_md5','p_halfyear'], 'od_brw_f12m')
    test_features = pd.merge(test_features, test_temp, on='fuid_md5', how='left')
    
    test_temp = groupby_std_pivot_Tool(test_p12M, ['fuid_md5','p_halfyear'], 'od_brw_f12m')
    test_features = pd.merge(test_features, test_temp, on='fuid_md5', how='left')
    
    
    train_features.to_csv(r'features/train_p12M_statistics.csv', index=False)
    test_features.to_csv(r'features/test_p12M_statistics.csv', index=False)
    
    print("p12M_statistics_ext finishing...")
    return train_features, test_features



###############################过去六个月订单行为汇总#############################
def p6M_info_ext(k, method):
    train_p6M = pd.read_csv(r'pro_data/train_p6M.csv', nrows=None)
    test_p6M = pd.read_csv(r'pro_data/test_p6M.csv', nrows=None)
    
    def p6M_sort_pyear_month(df):
        return df.sort_values('pyear_month')
    
    def select_last_k_1(df, k=k):
        temp = pd.DataFrame(df[-k:], columns=df.columns)
        return temp
    
    train_p6M = train_p6M.groupby('fuid_md5', as_index=False).apply(p6M_sort_pyear_month)
    test_p6M = test_p6M.groupby('fuid_md5', as_index=False).apply(p6M_sort_pyear_month)
    
    train_p6M = train_p6M.groupby('fuid_md5', as_index=False).apply(select_last_k_1)
    test_p6M = test_p6M.groupby('fuid_md5', as_index=False).apply(select_last_k_1)
    
    drop_colname = ['pyear_month','cyc_date','fcredit_update_time']
    train_p6M = train_p6M.drop(drop_colname, axis=1)
    test_p6M = test_p6M.drop(drop_colname, axis=1)
    
    if method == 'mean':
        train_p6M = train_p6M.groupby('fuid_md5', as_index=False).mean()
        test_p6M = test_p6M.groupby('fuid_md5', as_index=False).mean()
        
        cols = ['fuid_md5'] + [x+'_mean_%s' % k for x in train_p6M.columns[1:]]
        train_p6M.columns = cols
        cols = ['fuid_md5'] + [x+'_mean_%s' % k for x in test_p6M.columns[1:]]
        test_p6M.columns = cols
    
        train_p6M.to_csv(r'features/train_p6M_last_%s_%s.csv' % (k, method), index=False)
        test_p6M.to_csv(r'features/test_p6M_last_%s_%s.csv' % (k, method), index=False)
    
    if method == 'sum':
        train_p6M = train_p6M.groupby('fuid_md5', as_index=False).sum()
        test_p6M = test_p6M.groupby('fuid_md5', as_index=False).sum()
        
        cols = ['fuid_md5'] + [x+'_mean_%s' % k for x in train_p6M.columns[1:]]
        train_p6M.columns = cols
        cols = ['fuid_md5'] + [x+'_mean_%s' % k for x in test_p6M.columns[1:]]
        test_p6M.columns = cols
    
        train_p6M.to_csv(r'features/train_p6M_last_%s_%s.csv' % (k, method), index=False)
        test_p6M.to_csv(r'features/test_p6M_last_%s_%s.csv' % (k, method), index=False)
    
    
    print("p6M_info_ext_%s_%s finishing..." % (k, method))
    return train_p6M, test_p6M

###############################过去六个月订单行为（基于业务进行特征构造）#############################
#def p6M_info_ext(k):
#fopen_to_buy（剩余可用额度为正负的个数）
#futilization（大于1的次数）    
#fcredit_update_time（额度更新时间和注册审批时间和毕业时间的差值）    
#credit_limit（描述额度是否增长（unique））
def p6M_business_ext(k):
    train_p6M = pd.read_csv(r'pro_data/train_p6M.csv', nrows=None)
    test_p6M = pd.read_csv(r'pro_data/test_p6M.csv', nrows=None)
    
    def p6M_sort_pyear_month(df):
            return df.sort_values('pyear_month')
        
    def select_last_k_1(df, k=k):
        temp = pd.DataFrame(df[-k:], columns=df.columns)
        return temp
        
    train_p6M = train_p6M.groupby('fuid_md5', as_index=False).apply(p6M_sort_pyear_month)
    test_p6M = test_p6M.groupby('fuid_md5', as_index=False).apply(p6M_sort_pyear_month)
    
    train_p6M = train_p6M.groupby('fuid_md5', as_index=False).apply(select_last_k_1)
    test_p6M = test_p6M.groupby('fuid_md5', as_index=False).apply(select_last_k_1)

    
    train_p6M['fopen_to_buy'] = train_p6M['fopen_to_buy'].apply(lambda x: 1 if x<0  else 0)
    train_p6M_features = train_p6M.groupby('fuid_md5')['fopen_to_buy'].sum()\
                                          .reset_index()\
                                          .rename(columns={'fopen_to_buy':'fopen_to_buy_over1_sum'})
                                   
    train_p6M['futilization'] = train_p6M['futilization'].apply(lambda x: 1 if x>1  else 0)
    temp_features = train_p6M.groupby('fuid_md5')['futilization'].sum()\
                                     .reset_index()\
                                     .rename(columns={'futilization':'futilization_over1_sum'})
    train_p6M_features['futilization_over1_sum'] = temp_features['futilization_over1_sum']
    temp_features = train_p6M.groupby('fuid_md5')\
                                     .apply(lambda df: len(df['credit_limit'].unique()))\
                                     .reset_index()\
                                     .rename(columns={0:'credit_limit_unique'})
    train_p6M_features['credit_limit_unique'] = temp_features['credit_limit_unique']
    
    test_p6M['fopen_to_buy'] = test_p6M['fopen_to_buy'].apply(lambda x: 1 if x<0  else 0)
    test_p6M_features = test_p6M.groupby('fuid_md5')['fopen_to_buy'].sum()\
                                          .reset_index()\
                                          .rename(columns={'fopen_to_buy':'fopen_to_buy_over1_sum'})
    
    test_p6M['futilization'] = test_p6M['futilization'].apply(lambda x: 1 if x>1  else 0)
    temp_features = test_p6M.groupby('fuid_md5')['futilization'].sum()\
                                     .reset_index()\
                                     .rename(columns={'futilization':'futilization_over1_sum'})
    test_p6M_features['futilization_over1_sum'] = temp_features['futilization_over1_sum']
    
    temp_features = test_p6M.groupby('fuid_md5')\
                                     .apply(lambda df: len(df['credit_limit'].unique()))\
                                     .reset_index()\
                                     .rename(columns={0:'credit_limit_unique'})
    test_p6M_features['credit_limit_unique'] = temp_features['credit_limit_unique']
    
    cols = ['fuid_md5'] + [x+'_%s' % k for x in train_p6M_features.columns[1:]]
    train_p6M_features.columns = cols
    cols = ['fuid_md5'] + [x+'_%s' % k for x in test_p6M_features.columns[1:]]
    test_p6M_features.columns = cols
    
    train_p6M_features.to_csv(r'features/train_p6M_features_last_%s.csv' % k, index=False)
    test_p6M_features.to_csv(r'features/test_p6M_features_last_%s.csv' % k, index=False)
    
    print("p6M_business_ext %s finishing..." % k)
    return train_p6M_features, test_p6M_features


###############################过去六个月新增订单明细数据#############################
def od_in6m_info_ext():
    train_od_in6m_le = pd.read_csv(r'pro_data/train_od_in6m.csv', nrows=None, encoding='gbk')
    test_od_in6m_le = pd.read_csv(r'pro_data/test_od_in6m.csv', nrows=None, encoding='gbk')
            
    train_features = groupby_unique_Tool(train_od_in6m_le, 'fuid_md5', 'forder_type') 
    train_temp = groupby_unique_Tool(train_od_in6m_le, 'fuid_md5', 'fsub_order_type') 
    train_features = pd.merge(train_features, train_temp, on='fuid_md5', how='left')
    
    train_temp = groupby_unique_Tool(train_od_in6m_le, 'fuid_md5', 'forder_state') 
    train_features = pd.merge(train_features, train_temp, on='fuid_md5', how='left')
    
    train_temp = groupby_unique_Tool(train_od_in6m_le, 'fuid_md5', 'fsale_type') 
    train_features = pd.merge(train_features, train_temp, on='fuid_md5', how='left')
    
    train_temp = groupby_unique_Tool(train_od_in6m_le, 'fuid_md5', 'fsku_id') 
    train_features = pd.merge(train_features, train_temp, on='fuid_md5', how='left')
    
    train_temp = groupby_unique_Tool(train_od_in6m_le, 'fuid_md5', 'ffirstpay_fee_type') 
    train_features = pd.merge(train_features, train_temp, on='fuid_md5', how='left')
    
    train_temp = groupby_unique_Tool(train_od_in6m_le, 'fuid_md5', 'fmax_fq_num') 
    train_features = pd.merge(train_features, train_temp, on='fuid_md5', how='left')
    
    train_temp = groupby_sum_Tool(train_od_in6m_le, 'fuid_md5', 'ftotal_amount') 
    train_features = pd.merge(train_features, train_temp, on='fuid_md5', how='left')
    
    train_temp = groupby_sum_Tool(train_od_in6m_le, 'fuid_md5', 'ftotal_firstpay') 
    train_features = pd.merge(train_features, train_temp, on='fuid_md5', how='left')
    
    
    test_features = groupby_unique_Tool(test_od_in6m_le, 'fuid_md5', 'forder_type') 
    test_temp = groupby_unique_Tool(test_od_in6m_le, 'fuid_md5', 'fsub_order_type') 
    test_features = pd.merge(test_features, test_temp, on='fuid_md5', how='left')
    
    test_temp = groupby_unique_Tool(test_od_in6m_le, 'fuid_md5', 'forder_state') 
    test_features = pd.merge(test_features, test_temp, on='fuid_md5', how='left')
    
    test_temp = groupby_unique_Tool(test_od_in6m_le, 'fuid_md5', 'fsale_type') 
    test_features = pd.merge(test_features, test_temp, on='fuid_md5', how='left')
    
    test_temp = groupby_unique_Tool(test_od_in6m_le, 'fuid_md5', 'fsku_id') 
    test_features = pd.merge(test_features, test_temp, on='fuid_md5', how='left')
    
    test_temp = groupby_unique_Tool(test_od_in6m_le, 'fuid_md5', 'ffirstpay_fee_type') 
    test_features = pd.merge(test_features, test_temp, on='fuid_md5', how='left')
    
    test_temp = groupby_unique_Tool(test_od_in6m_le, 'fuid_md5', 'fmax_fq_num') 
    test_features = pd.merge(test_features, test_temp, on='fuid_md5', how='left')
    
    test_temp = groupby_sum_Tool(test_od_in6m_le, 'fuid_md5', 'ftotal_amount') 
    test_features = pd.merge(test_features, test_temp, on='fuid_md5', how='left')
    
    test_temp = groupby_sum_Tool(test_od_in6m_le, 'fuid_md5', 'ftotal_firstpay') 
    test_features = pd.merge(test_features, test_temp, on='fuid_md5', how='left')
    
    train_features.to_csv(r'features/train_od_in6m.csv', index=False)
    test_features.to_csv(r'features/test_od_in6m.csv', index=False)
    
    print("od_in6m_ext finishing...")
    return train_features, test_features

	
###############################过去六个月用户场景行为信息#############################
def login_scene_info_ext(k, method):
    train_login_scene = pd.read_csv(r'pro_data/train_login_scene.csv', nrows=None)
    test_login_scene = pd.read_csv(r'pro_data/test_login_scene.csv', nrows=None)
    
    def p6M_sort_pyear_month(df):
        return df.sort_values('pyear_month')
    
    def select_last_k_1(df, k=k):
        temp = pd.DataFrame(df[-k:], columns=df.columns)
        return temp
    
    train_login_scene = test_login_scene.groupby('fuid_md5', as_index=False).apply(p6M_sort_pyear_month)
    test_login_scene = test_login_scene.groupby('fuid_md5', as_index=False).apply(p6M_sort_pyear_month)
    
    train_login_scene = test_login_scene.groupby('fuid_md5', as_index=False).apply(select_last_k_1)
    test_login_scene = test_login_scene.groupby('fuid_md5', as_index=False).apply(select_last_k_1)
    
    drop_colname = ['pyear_month','cyc_date']
    train_login_scene = train_login_scene.drop(drop_colname, axis=1)
    test_login_scene = test_login_scene.drop(drop_colname, axis=1)
    
    if method == 'mean':
        train_login_scene = train_login_scene.groupby('fuid_md5', as_index=False).mean()
        test_login_scene = test_login_scene.groupby('fuid_md5', as_index=False).mean()
        
        cols = ['fuid_md5'] + [x+'_mean_%s' % k for x in train_login_scene.columns[1:]]
        train_login_scene.columns = cols
        cols = ['fuid_md5'] + [x+'_mean_%s' % k for x in test_login_scene.columns[1:]]
        test_login_scene.columns = cols
    
    if method == 'sum':
        train_login_scene = train_login_scene.groupby('fuid_md5', as_index=False).sum()
        test_login_scene = test_login_scene.groupby('fuid_md5', as_index=False).sum()
        
        cols = ['fuid_md5'] + [x+'_sum_%s' % k for x in train_login_scene.columns[1:]]
        train_login_scene.columns = cols
        cols = ['fuid_md5'] + [x+'_sum_%s' % k for x in test_login_scene.columns[1:]]
        test_login_scene.columns = cols
        
    train_login_scene.to_csv(r'features/train_login_scene_last_%s_%s.csv' % (k, method), index=False)
    test_login_scene.to_csv(r'features/test_login_scene_last_%s_%s.csv' % (k, method), index=False)
    
    print("login_scene_ext_%s_%s finishing..." % (k, method))
    return train_login_scene, train_login_scene


if __name__ == "__main__":
    ud_info_ext()
    p12M_info_ext()
    p6M_info_ext(1,'mean')
    p6M_info_ext(3,'mean')
    p6M_info_ext(6,'mean')
    p6M_info_ext(1,'sum')
    p6M_info_ext(3,'sum')
    p6M_info_ext(6,'sum')
    p6M_business_ext(1)
    p6M_business_ext(3)
    p6M_business_ext(6)
    login_scene_info_ext(1,'mean')
    login_scene_info_ext(3,'mean')
    login_scene_info_ext(6,'mean')
    login_scene_info_ext(3,'sum')
    login_scene_info_ext(6,'sum')
    od_in6m_info_ext()
    p12M_statistics_ext()
   
    print("finish...")








































