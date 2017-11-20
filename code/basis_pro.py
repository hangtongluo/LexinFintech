''# -*- coding: utf-8 -*-
"""
@author: Administrator
"""
import gc 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
from utils import LabelEncoderTool, ToDatetimeTool

'''
dep_mdl：训练目标文件
['fuid_md5', 'dep', 'actual_od_brw_f6m', 'actual_od_brw_1stm','actual_od_brw_2stm', 
'actual_od_brw_3stm', 'actual_od_brw_4stm','actual_od_brw_5stm', 'actual_od_brw_6stm']

login_scene_mdl：训练
['fuid', 'cyc_date', 'pyear_month', 'c_log_eqp_dist_cnt',
       'c_scene_tot_cnt', 'c_scene_dist_cnt', 'c_scene_mac_dist_cnt',
       'c_scene_chn_dist_cnt', 'c_scene_ip_dist_cnt',
       'c_scene_ipprov_dist_cnt', 'c_scene_ipcity_dist_cnt',
       'c_scene_appsys_dist_cnt', 'c_scene_wifi_dist_cnt',
       'c_scene_imei_dist_cnt', 'c_scene_tot_days', 'c_scene_reg_tot_cnt',
       'c_scene_dl_tot_cnt', 'c_scene_od_tot_cnt', 'c_scene_rp_tot_cnt',
       'c_scene_xgxx_tot_cnt', 'c_scene_plsp_tot_cnt', 'c_scene_sczl_tot_cnt',
       'c_scene_sh_tot_cnt', 'c_scene_pc_tot_cnt', 'c_scene_app_tot_cnt',
       'c_scene_h5_tot_cnt', 'c_scene_android_tot_cnt', 'c_scene_ios_tot_cnt',
       'c_scene_log_avg_dur', 'c_scene_log_max_dur', 'c_scene_log_min_dur',
       'c_scene_reg_avg_dur', 'c_scene_dl_avg_dur', 'c_scene_od_avg_dur',
       'c_scene_rp_avg_dur', 'c_scene_xgxx_avg_dur', 'c_scene_plsp_avg_dur',
       'c_scene_sczl_avg_dur', 'c_scene_sh_avg_dur', 'c_scene_reg_max_dur',
       'c_scene_dl_max_dur', 'c_scene_od_max_dur', 'c_scene_rp_max_dur',
       'c_scene_xgxx_max_dur', 'c_scene_plsp_max_dur', 'c_scene_sczl_max_dur',
       'c_scene_sh_max_dur', 'c_scene_reg_min_dur', 'c_scene_dl_min_dur',
       'c_scene_od_min_dur', 'c_scene_rp_min_dur', 'c_scene_xgxx_min_dur',
       'c_scene_plsp_min_dur', 'c_scene_sczl_min_dur', 'c_scene_sh_min_dur']

od_in6m_mdl：训练
['fuid_md5', 'forder_id_md5', 'faccount_time', 'forder_type',
       'fsub_order_type', 'forder_state', 'fsale_type', 'fsku_id',
       'fproduct_info', 'ftotal_amount', 'ftotal_firstpay',
       'ffirstpay_fee_type', 'fmax_fq_num']

p6M_mdl：训练
['fuid_md5', 'pyear_month', 'cyc_date', 'od_cnt', 'actual_od_cnt',
       'virtual_od_cnt', 'od_3c_cnt', 'od_bh_cnt', 'od_yl_cnt', 'od_xj_cnt',
       ...
       'acre_repay_od_cpt', 'foverdue_paying_day', 'foverdue_paying_cyc',
       'foverdue_payed_day', 'foverdue_payed_cyc', 'cpt_pymt', 'credit_limit',
       'fcredit_update_time', 'futilization', 'fopen_to_buy']

p12M_mdl：训练
['fuid_md5', 'pyear_month', 'cyc_date', 'od_brw_f12m']

ud_mdl：训练
['fuid_md5', 'fschoolarea_name_md5', 'fage', 'fsex', 'fis_entrance_exam',
       'fregister_time', 'fpocket_auth_time', 'fdomicile_provice',
       'fdomicile_city', 'fdomicile_area', 'sch_fprovince_name',
       'sch_fcity_name', 'sch_fregion_name', 'sch_fcompany_name', 'fstd_num',
       'fcollege_level', 'fcal_graduation', 'fauth_source_type']

'''  

########################################处理用户基本数据###############################################
def ud_info_pro():
    print("============处理用户基本数据==============")
    ud_mdl = pd.read_csv(r'data/lexin_train/ud_mdl.csv', low_memory=False, nrows=None)
    ud_offtime = pd.read_csv(r'data/lexin_test/ud_offtime.csv', low_memory=False, nrows=None)
    
    usecols = ['fschoolarea_name_md5', 'fdomicile_provice',
           'fdomicile_city', 'fdomicile_area', 'sch_fprovince_name',
           'sch_fcity_name', 'sch_fregion_name', 'sch_fcompany_name']
    train_ud_le, test_ud_le = LabelEncoderTool(ud_mdl, ud_offtime, usecols)
    
    usecols = ['fregister_time', 'fpocket_auth_time', 'fcal_graduation']
    train_ud_le = ToDatetimeTool(train_ud_le, usecols)
    test_ud_le = ToDatetimeTool(test_ud_le, usecols)
   
    train_ud_le.to_csv(r'pro_data/train_ud.csv', index=False)
    test_ud_le.to_csv(r'pro_data/test_ud.csv', index=False)
    
    
    
########################################过去12个月月度订单金额###############################################
def p12M_pro():
    print("============过去12个月月度订单金额==============")
    p12M_mdl = pd.read_csv(r'data/lexin_train/p12M_mdl.csv', low_memory=False, nrows=None)
    p12M_offtime = pd.read_csv(r'data/lexin_test/p12M_offtime.csv', low_memory=False, nrows=None)
    
    usecols = ['pyear_month']
    train_p12M = ToDatetimeTool(p12M_mdl, usecols)
    test_p12M = ToDatetimeTool(p12M_offtime, usecols) 

    train_p12M.to_csv(r'pro_data/train_p12M.csv', index=False)
    test_p12M.to_csv(r'pro_data/test_p12M.csv', index=False)



########################################过去六个月订单行为汇总###############################################
def p6M_pro():
    print("============过去六个月订单行为汇总==============")    
    p6M_mdl = pd.read_csv(r'data/lexin_train/p6M_mdl.csv', low_memory=False, nrows=None)
    p6M_offtime = pd.read_csv(r'data/lexin_test/p6M_offtime.csv', low_memory=False, nrows=None)
    
    usecols = ['pyear_month','fcredit_update_time']
    train_p6M = ToDatetimeTool(p6M_mdl, usecols)
    test_p6M = ToDatetimeTool(p6M_offtime, usecols)

    #汇总一些相似字段（当前月新建账单总数）
    print(train_p6M.shape, test_p6M.shape)
    
    ''''od_cnt', 'actual_od_cnt', 'virtual_od_cnt', 'od_3c_cnt', 'od_bh_cnt', 'od_yl_cnt',
       'od_xj_cnt', 'od_ptsh_cnt', 'od_zdfq_cnt', 'od_xssh_cnt', 'od_zdyq_cnt', 'od_lh_new_cnt','''
       
    usecols = ['actual_od_cnt', 'virtual_od_cnt', 'od_3c_cnt', 'od_bh_cnt', 'od_yl_cnt',\
       'od_xj_cnt', 'od_ptsh_cnt', 'od_zdfq_cnt', 'od_xssh_cnt', 'od_zdyq_cnt', 'od_lh_new_cnt']
    train_p6M['od_cnt_sum'] = train_p6M[usecols].sum(axis=1)
#    train_p6M = train_p6M.drop(usecols, axis=1)
    test_p6M['od_cnt_sum'] = test_p6M[usecols].sum(axis=1)
#    test_p6M = test_p6M.drop(usecols, axis=1)
    
    print(train_p6M.shape, test_p6M.shape)
    
    #汇总一些相似字段（历史存量创建账单总数）
    ''''cumu_od_cnt', 'cumu_actual_od_cnt', 'cumu_virtual_od_cnt', 'cumu_od_3c_cnt', 'cumu_od_bh_cnt',
       'cumu_od_yl_cnt', 'cumu_od_xj_cnt', 'cumu_od_ptsh_cnt','cumu_od_zdfq_cnt', 'cumu_od_xssh_cnt', 
       'cumu_od_zdyq_cnt', 'cumu_od_lh_new_cnt','''

    usecols = ['cumu_actual_od_cnt', 'cumu_virtual_od_cnt', 'cumu_od_3c_cnt', 'cumu_od_bh_cnt',\
               'cumu_od_yl_cnt', 'cumu_od_xj_cnt', 'cumu_od_ptsh_cnt','cumu_od_zdfq_cnt', 'cumu_od_xssh_cnt', \
               'cumu_od_zdyq_cnt', 'cumu_od_lh_new_cnt']
    train_p6M['cumu_od_cnt_sum'] = train_p6M[usecols].sum(axis=1)
#    train_p6M = train_p6M.drop(usecols, axis=1)
    test_p6M['cumu_od_cnt_sum'] = test_p6M[usecols].sum(axis=1)
#    test_p6M = test_p6M.drop(usecols, axis=1)
    
    print(train_p6M.shape, test_p6M.shape)
    
    
    #汇总一些相似字段（当前观测月新建订单金额总数）
    ''''od_brw', 'actual_od_brw','virtual_od_brw', 'od_3c_brw', 'od_bh_brw', 'od_yl_brw',
       'od_xj_brw', 'od_ptsh_brw', 'od_zdfq_brw', 'od_xssh_brw', 'od_zdyq_brw', 'od_lh_new_brw','''

    usecols = ['actual_od_brw','virtual_od_brw', 'od_3c_brw', 'od_bh_brw', 'od_yl_brw',\
               'od_xj_brw', 'od_ptsh_brw', 'od_zdfq_brw', 'od_xssh_brw', 'od_zdyq_brw', 'od_lh_new_brw']
    train_p6M['od_brw_sum'] = train_p6M[usecols].sum(axis=1)
#    train_p6M = train_p6M.drop(usecols, axis=1)
    test_p6M['od_brw_sum'] = test_p6M[usecols].sum(axis=1)
#    test_p6M = test_p6M.drop(usecols, axis=1)
    
    print(train_p6M.shape, test_p6M.shape)
    
    
    #汇总一些相似字段（历史存量创建订单金额总数）
    ''''cumu_od_brw', 'cumu_actual_od_brw', 'cumu_virtual_od_brw', 'cumu_od_3c_brw', 'cumu_od_bh_brw', 'cumu_od_yl_brw',
    'cumu_od_xj_brw', 'cumu_od_ptsh_brw','cumu_od_zdfq_brw', 'cumu_od_xssh_brw', 'cumu_od_zdyq_brw', 'cumu_od_lh_new_brw','''

    usecols = ['cumu_actual_od_brw', 'cumu_virtual_od_brw', 'cumu_od_3c_brw', 'cumu_od_bh_brw', 'cumu_od_yl_brw',\
               'cumu_od_xj_brw', 'cumu_od_ptsh_brw','cumu_od_zdfq_brw', 'cumu_od_xssh_brw', 'cumu_od_zdyq_brw', 'cumu_od_lh_new_brw']
    train_p6M['cumu_od_brw_sum'] = train_p6M[usecols].sum(axis=1)
#    train_p6M = train_p6M.drop(usecols, axis=1)
    test_p6M['cumu_od_brw_sum'] = test_p6M[usecols].sum(axis=1)
#    test_p6M = test_p6M.drop(usecols, axis=1)
    
    print(train_p6M.shape, test_p6M.shape)


    #汇总一些相似字段（截止到当前应还款日的已还本金总和）
    ''''payed_capital', 'payed_actual_capital','payed_virtual_capital', 'payed_3c_capital', 'payed_bh_capital',
       'payed_yl_capital', 'payed_xj_capital', 'payed_ptsh_capital','payed_zdfq_capital', 'payed_xssh_capital', 'payed_zdyq_capital',
       'payed_lh_new_capital','''

    usecols = ['payed_actual_capital','payed_virtual_capital', 'payed_3c_capital', 'payed_bh_capital', 'payed_yl_capital', \
    'payed_xj_capital', 'payed_ptsh_capital','payed_zdfq_capital', 'payed_xssh_capital', 'payed_zdyq_capital', 'payed_lh_new_capital']
    train_p6M['payed_capital_sum'] = train_p6M[usecols].sum(axis=1)
#    train_p6M = train_p6M.drop(usecols, axis=1)
    test_p6M['payed_capital_sum'] = test_p6M[usecols].sum(axis=1)
#    test_p6M = test_p6M.drop(usecols, axis=1)
    
    print(train_p6M.shape, test_p6M.shape)


    #汇总一些相似字段（截止到当前应还款日的已还月服务费总和）
    ''''payed_mon_fee', 'payed_3c_mon_fee','payed_bh_mon_fee', 'payed_yl_mon_fee', 'payed_xj_mon_fee', 'payed_ptsh_mon_fee', 
        'payed_zdfq_mon_fee', 'payed_xssh_mon_fee', 'payed_zdyq_mon_fee', 'payed_lh_new_mon_fee', 'payed_tot_fee',
       'payed_3c_tot_fee', 'payed_bh_tot_fee', 'payed_yl_tot_fee', 'payed_xj_tot_fee', 'payed_ptsh_tot_fee', 'payed_zdfq_tot_fee',
       'payed_xssh_tot_fee', 'payed_zdyq_tot_fee', 'payed_lh_new_tot_fee','''

    usecols = ['payed_3c_mon_fee','payed_bh_mon_fee', 'payed_yl_mon_fee', 'payed_xj_mon_fee', 'payed_ptsh_mon_fee',\
               'payed_zdfq_mon_fee', 'payed_xssh_mon_fee', 'payed_zdyq_mon_fee', 'payed_lh_new_mon_fee', 'payed_tot_fee',\
               'payed_3c_tot_fee', 'payed_bh_tot_fee', 'payed_yl_tot_fee', 'payed_xj_tot_fee', 'payed_ptsh_tot_fee', 'payed_zdfq_tot_fee',\
               'payed_xssh_tot_fee', 'payed_zdyq_tot_fee', 'payed_lh_new_tot_fee']
    train_p6M['payed_mon_fee_sum'] = train_p6M[usecols].sum(axis=1)
#    train_p6M = train_p6M.drop(usecols, axis=1)
    test_p6M['payed_mon_fee_sum'] = test_p6M[usecols].sum(axis=1)
#    test_p6M = test_p6M.drop(usecols, axis=1)
    
    print(train_p6M.shape, test_p6M.shape)
    
   
    #汇总一些相似字段（截止到当前应还款日的待还本金总和）
    ''''bal', 'ds3c_bal', 'bh_bal', 'yl_bal', 'xj_bal', 'ptsh_bal','zdfq_bal', 'xssh_bal', 'zdyq_bal', 'lh_new_bal','''

    usecols = ['ds3c_bal', 'bh_bal', 'yl_bal', 'xj_bal', 'ptsh_bal','zdfq_bal', 'xssh_bal', 'zdyq_bal', 'lh_new_bal']
    train_p6M['bal_sum'] = train_p6M[usecols].sum(axis=1)
#    train_p6M = train_p6M.drop(usecols, axis=1)
    test_p6M['bal_sum'] = test_p6M[usecols].sum(axis=1)
#    test_p6M = test_p6M.drop(usecols, axis=1)
    
    print(train_p6M.shape, test_p6M.shape)
    
    #汇总一些相似字段（截止到当前应还款日的应还服务费总和）
    ''''paying_mon_fee', 'ds3c_paying_mon_fee', 'bh_paying_mon_fee', 'yl_paying_mon_fee', 'xj_paying_mon_fee', 
        'ptsh_paying_mon_fee', 'zdfq_paying_mon_fee', 'xssh_paying_mon_fee', 'zdyq_paying_mon_fee', 'lh_new_paying_mon_fee', 
        'paying_tot_fee', 'ds3c_paying_tot_fee', 'bh_paying_tot_fee', 'yl_paying_tot_fee', 'xj_paying_tot_fee',
       'ptsh_paying_tot_fee', 'zdfq_paying_tot_fee', 'xssh_paying_tot_fee', 'zdyq_paying_tot_fee', 'lh_new_paying_tot_fee','''

    usecols = ['ds3c_paying_mon_fee', 'bh_paying_mon_fee', 'yl_paying_mon_fee', 'xj_paying_mon_fee',\
               'ptsh_paying_mon_fee', 'zdfq_paying_mon_fee', 'xssh_paying_mon_fee', 'zdyq_paying_mon_fee', 'lh_new_paying_mon_fee',\
               'paying_tot_fee', 'ds3c_paying_tot_fee', 'bh_paying_tot_fee', 'yl_paying_tot_fee', 'xj_paying_tot_fee',\
               'ptsh_paying_tot_fee', 'zdfq_paying_tot_fee', 'xssh_paying_tot_fee', 'zdyq_paying_tot_fee', 'lh_new_paying_tot_fee']
    train_p6M['paying_mon_fee_sum'] = train_p6M[usecols].sum(axis=1)
#    train_p6M = train_p6M.drop(usecols, axis=1)
    test_p6M['paying_mon_fee_sum'] = test_p6M[usecols].sum(axis=1)
#    test_p6M = test_p6M.drop(usecols, axis=1)
    
    print(train_p6M.shape, test_p6M.shape)
    
    #当月花费和还款是超过信誉度多少
    train_p6M['credit_fopen_cpt'] = train_p6M['credit_limit'] - train_p6M['fopen_to_buy'] - train_p6M['cpt_pymt']
    test_p6M['credit_fopen_cpt'] = test_p6M['credit_limit'] - test_p6M['fopen_to_buy'] - test_p6M['cpt_pymt']
    
    #截止到当前还款日应全部还完订单金额和已全部换完订单金额，看还有多少没还
    train_p6M['paying_payed_complete_od_brw'] = train_p6M['paying_complete_od_brw'] - train_p6M['payed_complete_od_brw']
    test_p6M['paying_payed_complete_od_brw'] = test_p6M['paying_complete_od_brw'] - test_p6M['payed_complete_od_brw']
        
    #截止到当前还款日应全部还完订单金额和已全部换完订单数，看还有多少没还
    train_p6M['paying_payed_complete_od_cnt'] = train_p6M['paying_complete_od_cnt'] - train_p6M['payed_complete_od_cnt']
    test_p6M['paying_payed_complete_od_cnt'] = test_p6M['paying_complete_od_cnt'] - test_p6M['payed_complete_od_cnt']
       
	
    train_p6M.to_csv(r'pro_data/train_p6M.csv', index=False)
    test_p6M.to_csv(r'pro_data/test_p6M.csv', index=False)
    


########################################过去六个月新增订单明细数据###############################################
def od_in6m_pro():
    print("============过去六个月新增订单明细数据==============")
    od_in6m_mdl = pd.read_csv(r'data/lexin_train/od_in6m_mdl.csv', low_memory=False, nrows=None)
    od_in6m_offtime = pd.read_csv(r'data/lexin_test/od_in6m_offtime.csv', low_memory=False, nrows=None)
    
    usecols = ['faccount_time']
    train_od_in6m = ToDatetimeTool(od_in6m_mdl, usecols)
    test_od_in6m = ToDatetimeTool(od_in6m_offtime, usecols)
    
    usecols = ['forder_id_md5', 'fsku_id']
    train_od_in6m_le, test_od_in6m_le = LabelEncoderTool(train_od_in6m, test_od_in6m, usecols)
    
    train_od_in6m_le.to_csv(r'pro_data/train_od_in6m.csv', index=False)
    test_od_in6m_le.to_csv(r'pro_data/test_od_in6m.csv', index=False)


########################################过去六个月用户场景行为信息###############################################
def login_scene_pro():
    print("============过去六个月用户场景行为信息==============")
    login_scene_mdl = pd.read_csv(r'data/lexin_train/login_scene_mdl.csv', low_memory=False, nrows=None)
    login_scene_offtime = pd.read_csv(r'data/lexin_test/login_scene_offtime.csv', low_memory=False, nrows=None)

    usecols = ['pyear_month']
    train_login_scene = ToDatetimeTool(login_scene_mdl, usecols)
    test_login_scene = ToDatetimeTool(login_scene_offtime, usecols)

    train_login_scene.to_csv(r'pro_data/train_login_scene.csv', index=False)
    test_login_scene.to_csv(r'pro_data/test_login_scene.csv', index=False)



def dep_pro():
    print("============train_dep_target==============")
    dep_mdl = pd.read_csv(r'data/lexin_train/dep_mdl.csv', low_memory=False, nrows=None)
    dep_mdl.to_csv(r'pro_data/train_dep_target.csv', index=False)



if __name__ == "__main__":
    ud_info_pro()
    p12M_pro()
    p6M_pro()
    od_in6m_pro()
    login_scene_pro()
    dep_pro()
    print('finish...')





















