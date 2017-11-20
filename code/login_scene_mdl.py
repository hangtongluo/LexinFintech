# -- coding: utf-8 --
import pandas as pd
import numpy as np

login_scene_mdl = pd.read_csv(r"data\lexin_train\login_scene_mdl.csv")

import datetime as dt
login_scene_mdl['cyc_date'] = pd.to_datetime(login_scene_mdl['cyc_date'],format='%Y-%m-%d %H:%M:%S')
login_scene_mdl['cyc_date']=[i.month for i in login_scene_mdl["cyc_date"]]

login_scene_mdl=login_scene_mdl.drop('pyear_month',axis=1)

#应该先弄好每个人有的六行数据的每个特征的最大值，最小值，均值，方差，中位数，众值
login_scene_mdl_mean = login_scene_mdl.groupby(['fuid_md5']).agg(np.mean)
login_scene_mdl_max = login_scene_mdl.groupby(['fuid_md5']).agg(np.max)
login_scene_mdl_min = login_scene_mdl.groupby(['fuid_md5']).agg(np.min)
login_scene_mdl_std = login_scene_mdl.groupby(['fuid_md5']).agg(np.std)
login_scene_mdl_median = login_scene_mdl.groupby(['fuid_md5']).agg(np.median)

login_scene_mdl_mean = login_scene_mdl_mean.reset_index()
login_scene_mdl_max = login_scene_mdl_max.reset_index()
login_scene_mdl_min = login_scene_mdl_min.reset_index()
login_scene_mdl_std = login_scene_mdl_std.reset_index()
login_scene_mdl_median = login_scene_mdl_median.reset_index()

login_scene_mdl_mean = login_scene_mdl_mean.drop('cyc_date',axis=1)
login_scene_mdl_max = login_scene_mdl_max.drop('cyc_date',axis=1)
login_scene_mdl_min = login_scene_mdl_min.drop('cyc_date',axis=1)
login_scene_mdl_std = login_scene_mdl_std.drop('cyc_date',axis=1)
login_scene_mdl_median = login_scene_mdl_median.drop('cyc_date',axis=1)

#将均值，最大值，最小值，方差，中值拼接起来，应该是53*5+1列数据
login_scene_sta = pd.merge(login_scene_mdl_mean,login_scene_mdl_max,how='left',on ='fuid_md5',suffixes=['_mean', '_max'])
login_scene_sta = pd.merge(login_scene_sta,login_scene_mdl_min,how='left',on ='fuid_md5',suffixes=['', '_min'])
login_scene_sta = pd.merge(login_scene_sta,login_scene_mdl_std,how='left',on ='fuid_md5',suffixes=['', '_std'])
login_scene_sta = pd.merge(login_scene_sta,login_scene_mdl_median,how='left',on ='fuid_md5',suffixes=['', '_median'])

#获取5,6,7,8,9,10月份数据
login_scene_mdl_10 = login_scene_mdl[login_scene_mdl['cyc_date']==10]
login_scene_mdl_10 = login_scene_mdl_10.drop('cyc_date',axis=1)

login_scene_mdl_9 = login_scene_mdl[login_scene_mdl['cyc_date']==9]
login_scene_mdl_9 = login_scene_mdl_9.drop('cyc_date',axis=1)

login_scene_mdl_8 = login_scene_mdl[login_scene_mdl['cyc_date']==8]
login_scene_mdl_8 = login_scene_mdl_8.drop('cyc_date',axis=1)

login_scene_mdl_7 = login_scene_mdl[login_scene_mdl['cyc_date']==7]
login_scene_mdl_7 = login_scene_mdl_7.drop('cyc_date',axis=1)

login_scene_mdl_6 = login_scene_mdl[login_scene_mdl['cyc_date']==6]
login_scene_mdl_6 = login_scene_mdl_6.drop('cyc_date',axis=1)

login_scene_mdl_5 = login_scene_mdl[login_scene_mdl['cyc_date']==5]
login_scene_mdl_5 = login_scene_mdl_5.drop('cyc_date',axis=1)

#将月份数据拼接起来
login_scene_mon = pd.merge(login_scene_mdl_10,login_scene_mdl_9,how='left',on ='fuid_md5',suffixes=['_10', '_9'])
login_scene_mon = pd.merge(login_scene_mon,login_scene_mdl_8,how='left',on ='fuid_md5',suffixes=['', '_8'])
login_scene_mon = pd.merge(login_scene_mon,login_scene_mdl_7,how='left',on ='fuid_md5',suffixes=['', '_7'])
login_scene_mon = pd.merge(login_scene_mon,login_scene_mdl_6,how='left',on ='fuid_md5',suffixes=['', '_6'])
login_scene_mon = pd.merge(login_scene_mon,login_scene_mdl_5,how='left',on ='fuid_md5',suffixes=['', '_5'])

#再将上面两种数据拼接，一种是统计数据，一种是每月的数据
login_scene_mdl = pd.merge(login_scene_mon,login_scene_sta,how='left',on ='fuid_md5')

#接下来我要处理：当前观测月，场景分别为注册，登陆，，，的次数，/总次数
#10月
login_scene_mdl['c_scene_reg_tot_fre_10'] = login_scene_mdl['c_scene_reg_tot_cnt_10']/login_scene_mdl['c_scene_tot_cnt_10']
login_scene_mdl['c_scene_dl_tot_fre_10'] = login_scene_mdl['c_scene_dl_tot_cnt_10']/login_scene_mdl['c_scene_tot_cnt_10']
login_scene_mdl['c_scene_od_tot_fre_10'] = login_scene_mdl['c_scene_od_tot_cnt_10']/login_scene_mdl['c_scene_tot_cnt_10']
login_scene_mdl['c_scene_rp_tot_fre_10'] = login_scene_mdl['c_scene_rp_tot_cnt_10']/login_scene_mdl['c_scene_tot_cnt_10']
login_scene_mdl['c_scene_xgxx_tot_fre_10'] = login_scene_mdl['c_scene_xgxx_tot_cnt_10']/login_scene_mdl['c_scene_tot_cnt_10']
login_scene_mdl['c_scene_plsp_tot_fre_10'] = login_scene_mdl['c_scene_plsp_tot_cnt_10']/login_scene_mdl['c_scene_tot_cnt_10']
login_scene_mdl['c_scene_sczl_tot_fre_10'] = login_scene_mdl['c_scene_sczl_tot_cnt_10']/login_scene_mdl['c_scene_tot_cnt_10']
login_scene_mdl['c_scene_sh_tot_fre_10'] = login_scene_mdl['c_scene_sh_tot_cnt_10']/login_scene_mdl['c_scene_tot_cnt_10']

#9月
login_scene_mdl['c_scene_reg_tot_fre_9'] = login_scene_mdl['c_scene_reg_tot_cnt_9']/login_scene_mdl['c_scene_tot_cnt_9']
login_scene_mdl['c_scene_dl_tot_fre_9'] = login_scene_mdl['c_scene_dl_tot_cnt_9']/login_scene_mdl['c_scene_tot_cnt_9']
login_scene_mdl['c_scene_od_tot_fre_9'] = login_scene_mdl['c_scene_od_tot_cnt_9']/login_scene_mdl['c_scene_tot_cnt_9']
login_scene_mdl['c_scene_rp_tot_fre_9'] = login_scene_mdl['c_scene_rp_tot_cnt_9']/login_scene_mdl['c_scene_tot_cnt_9']
login_scene_mdl['c_scene_xgxx_tot_fre_9'] = login_scene_mdl['c_scene_xgxx_tot_cnt_9']/login_scene_mdl['c_scene_tot_cnt_9']
login_scene_mdl['c_scene_plsp_tot_fre_9'] = login_scene_mdl['c_scene_plsp_tot_cnt_9']/login_scene_mdl['c_scene_tot_cnt_9']
login_scene_mdl['c_scene_sczl_tot_fre_9'] = login_scene_mdl['c_scene_sczl_tot_cnt_9']/login_scene_mdl['c_scene_tot_cnt_9']
login_scene_mdl['c_scene_sh_tot_fre_9'] = login_scene_mdl['c_scene_sh_tot_cnt_9']/login_scene_mdl['c_scene_tot_cnt_9']

#8月
login_scene_mdl['c_scene_reg_tot_fre_x'] = login_scene_mdl['c_scene_reg_tot_cnt_x']/login_scene_mdl['c_scene_tot_cnt_x']
login_scene_mdl['c_scene_dl_tot_fre_x'] = login_scene_mdl['c_scene_dl_tot_cnt_x']/login_scene_mdl['c_scene_tot_cnt_x']
login_scene_mdl['c_scene_od_tot_fre_x'] = login_scene_mdl['c_scene_od_tot_cnt_x']/login_scene_mdl['c_scene_tot_cnt_x']
login_scene_mdl['c_scene_rp_tot_fre_x'] = login_scene_mdl['c_scene_rp_tot_cnt_x']/login_scene_mdl['c_scene_tot_cnt_x']
login_scene_mdl['c_scene_xgxx_tot_fre_x'] = login_scene_mdl['c_scene_xgxx_tot_cnt_x']/login_scene_mdl['c_scene_tot_cnt_x']
login_scene_mdl['c_scene_plsp_tot_fre_x'] = login_scene_mdl['c_scene_plsp_tot_cnt_x']/login_scene_mdl['c_scene_tot_cnt_x']
login_scene_mdl['c_scene_sczl_tot_fre_x'] = login_scene_mdl['c_scene_sczl_tot_cnt_x']/login_scene_mdl['c_scene_tot_cnt_x']
login_scene_mdl['c_scene_sh_tot_fre_x'] = login_scene_mdl['c_scene_sh_tot_cnt_x']/login_scene_mdl['c_scene_tot_cnt_x']

#7月
login_scene_mdl['c_scene_reg_tot_fre_7'] = login_scene_mdl['c_scene_reg_tot_cnt_7']/login_scene_mdl['c_scene_tot_cnt_7']
login_scene_mdl['c_scene_dl_tot_fre_7'] = login_scene_mdl['c_scene_dl_tot_cnt_7']/login_scene_mdl['c_scene_tot_cnt_7']
login_scene_mdl['c_scene_od_tot_fre_7'] = login_scene_mdl['c_scene_od_tot_cnt_7']/login_scene_mdl['c_scene_tot_cnt_7']
login_scene_mdl['c_scene_rp_tot_fre_7'] = login_scene_mdl['c_scene_rp_tot_cnt_7']/login_scene_mdl['c_scene_tot_cnt_7']
login_scene_mdl['c_scene_xgxx_tot_fre_7'] = login_scene_mdl['c_scene_xgxx_tot_cnt_7']/login_scene_mdl['c_scene_tot_cnt_7']
login_scene_mdl['c_scene_plsp_tot_fre_7'] = login_scene_mdl['c_scene_plsp_tot_cnt_7']/login_scene_mdl['c_scene_tot_cnt_7']
login_scene_mdl['c_scene_sczl_tot_fre_7'] = login_scene_mdl['c_scene_sczl_tot_cnt_7']/login_scene_mdl['c_scene_tot_cnt_7']
login_scene_mdl['c_scene_sh_tot_fre_7'] = login_scene_mdl['c_scene_sh_tot_cnt_7']/login_scene_mdl['c_scene_tot_cnt_7']

#6月
login_scene_mdl['c_scene_reg_tot_fre_6'] = login_scene_mdl['c_scene_reg_tot_cnt_6']/login_scene_mdl['c_scene_tot_cnt_6']
login_scene_mdl['c_scene_dl_tot_fre_6'] = login_scene_mdl['c_scene_dl_tot_cnt_6']/login_scene_mdl['c_scene_tot_cnt_6']
login_scene_mdl['c_scene_od_tot_fre_6'] = login_scene_mdl['c_scene_od_tot_cnt_6']/login_scene_mdl['c_scene_tot_cnt_6']
login_scene_mdl['c_scene_rp_tot_fre_6'] = login_scene_mdl['c_scene_rp_tot_cnt_6']/login_scene_mdl['c_scene_tot_cnt_6']
login_scene_mdl['c_scene_xgxx_tot_fre_6'] = login_scene_mdl['c_scene_xgxx_tot_cnt_6']/login_scene_mdl['c_scene_tot_cnt_6']
login_scene_mdl['c_scene_plsp_tot_fre_6'] = login_scene_mdl['c_scene_plsp_tot_cnt_6']/login_scene_mdl['c_scene_tot_cnt_6']
login_scene_mdl['c_scene_sczl_tot_fre_6'] = login_scene_mdl['c_scene_sczl_tot_cnt_6']/login_scene_mdl['c_scene_tot_cnt_6']
login_scene_mdl['c_scene_sh_tot_fre_6'] = login_scene_mdl['c_scene_sh_tot_cnt_6']/login_scene_mdl['c_scene_tot_cnt_6']

#5月
login_scene_mdl['c_scene_reg_tot_fre_5'] = login_scene_mdl['c_scene_reg_tot_cnt_5']/login_scene_mdl['c_scene_tot_cnt_5']
login_scene_mdl['c_scene_dl_tot_fre_5'] = login_scene_mdl['c_scene_dl_tot_cnt_5']/login_scene_mdl['c_scene_tot_cnt_5']
login_scene_mdl['c_scene_od_tot_fre_5'] = login_scene_mdl['c_scene_od_tot_cnt_5']/login_scene_mdl['c_scene_tot_cnt_5']
login_scene_mdl['c_scene_rp_tot_fre_5'] = login_scene_mdl['c_scene_rp_tot_cnt_5']/login_scene_mdl['c_scene_tot_cnt_5']
login_scene_mdl['c_scene_xgxx_tot_fre_5'] = login_scene_mdl['c_scene_xgxx_tot_cnt_5']/login_scene_mdl['c_scene_tot_cnt_5']
login_scene_mdl['c_scene_plsp_tot_fre_5'] = login_scene_mdl['c_scene_plsp_tot_cnt_5']/login_scene_mdl['c_scene_tot_cnt_5']
login_scene_mdl['c_scene_sczl_tot_fre_5'] = login_scene_mdl['c_scene_sczl_tot_cnt_5']/login_scene_mdl['c_scene_tot_cnt_5']
login_scene_mdl['c_scene_sh_tot_fre_5'] = login_scene_mdl['c_scene_sh_tot_cnt_5']/login_scene_mdl['c_scene_tot_cnt_5']

#接下来我要弄一个不同设备的次数/总次数
login_scene_mdl['c_scene_pc_tot_fre_10'] = login_scene_mdl['c_scene_pc_tot_cnt_10']/login_scene_mdl['c_scene_tot_cnt_10']
login_scene_mdl['c_scene_app_tot_fre_10'] = login_scene_mdl['c_scene_app_tot_cnt_10']/login_scene_mdl['c_scene_tot_cnt_10']
login_scene_mdl['c_scene_h5_tot_fre_10'] = login_scene_mdl['c_scene_h5_tot_cnt_10']/login_scene_mdl['c_scene_tot_cnt_10']
login_scene_mdl['c_scene_android_tot_fre_10'] = login_scene_mdl['c_scene_android_tot_cnt_10']/login_scene_mdl['c_scene_tot_cnt_10']
login_scene_mdl['c_scene_ios_tot_fre_10'] = login_scene_mdl['c_scene_ios_tot_cnt_10']/login_scene_mdl['c_scene_tot_cnt_10']

login_scene_mdl['c_scene_pc_tot_fre_9'] = login_scene_mdl['c_scene_pc_tot_cnt_9']/login_scene_mdl['c_scene_tot_cnt_9']
login_scene_mdl['c_scene_app_tot_fre_9'] = login_scene_mdl['c_scene_app_tot_cnt_9']/login_scene_mdl['c_scene_tot_cnt_9']
login_scene_mdl['c_scene_h5_tot_fre_9'] = login_scene_mdl['c_scene_h5_tot_cnt_9']/login_scene_mdl['c_scene_tot_cnt_9']
login_scene_mdl['c_scene_android_tot_fre_9'] = login_scene_mdl['c_scene_android_tot_cnt_9']/login_scene_mdl['c_scene_tot_cnt_9']
login_scene_mdl['c_scene_ios_tot_fre_9'] = login_scene_mdl['c_scene_ios_tot_cnt_9']/login_scene_mdl['c_scene_tot_cnt_9']

login_scene_mdl['c_scene_pc_tot_fre_x'] = login_scene_mdl['c_scene_pc_tot_cnt_x']/login_scene_mdl['c_scene_tot_cnt_x']
login_scene_mdl['c_scene_app_tot_fre_x'] = login_scene_mdl['c_scene_app_tot_cnt_x']/login_scene_mdl['c_scene_tot_cnt_x']
login_scene_mdl['c_scene_h5_tot_fre_x'] = login_scene_mdl['c_scene_h5_tot_cnt_x']/login_scene_mdl['c_scene_tot_cnt_x']
login_scene_mdl['c_scene_android_tot_fre_x'] = login_scene_mdl['c_scene_android_tot_cnt_x']/login_scene_mdl['c_scene_tot_cnt_x']
login_scene_mdl['c_scene_ios_tot_fre_x'] = login_scene_mdl['c_scene_ios_tot_cnt_x']/login_scene_mdl['c_scene_tot_cnt_x']

login_scene_mdl['c_scene_pc_tot_fre_7'] = login_scene_mdl['c_scene_pc_tot_cnt_7']/login_scene_mdl['c_scene_tot_cnt_7']
login_scene_mdl['c_scene_app_tot_fre_7'] = login_scene_mdl['c_scene_app_tot_cnt_7']/login_scene_mdl['c_scene_tot_cnt_7']
login_scene_mdl['c_scene_h5_tot_fre_7'] = login_scene_mdl['c_scene_h5_tot_cnt_7']/login_scene_mdl['c_scene_tot_cnt_7']
login_scene_mdl['c_scene_android_tot_fre_7'] = login_scene_mdl['c_scene_android_tot_cnt_7']/login_scene_mdl['c_scene_tot_cnt_7']
login_scene_mdl['c_scene_ios_tot_fre_7'] = login_scene_mdl['c_scene_ios_tot_cnt_7']/login_scene_mdl['c_scene_tot_cnt_7']

login_scene_mdl['c_scene_pc_tot_fre_6'] = login_scene_mdl['c_scene_pc_tot_cnt_6']/login_scene_mdl['c_scene_tot_cnt_6']
login_scene_mdl['c_scene_app_tot_fre_6'] = login_scene_mdl['c_scene_app_tot_cnt_6']/login_scene_mdl['c_scene_tot_cnt_6']
login_scene_mdl['c_scene_h5_tot_fre_6'] = login_scene_mdl['c_scene_h5_tot_cnt_6']/login_scene_mdl['c_scene_tot_cnt_6']
login_scene_mdl['c_scene_android_tot_fre_6'] = login_scene_mdl['c_scene_android_tot_cnt_6']/login_scene_mdl['c_scene_tot_cnt_6']
login_scene_mdl['c_scene_ios_tot_fre_6'] = login_scene_mdl['c_scene_ios_tot_cnt_6']/login_scene_mdl['c_scene_tot_cnt_6']

login_scene_mdl['c_scene_pc_tot_fre_5'] = login_scene_mdl['c_scene_pc_tot_cnt_5']/login_scene_mdl['c_scene_tot_cnt_5']
login_scene_mdl['c_scene_app_tot_fre_5'] = login_scene_mdl['c_scene_app_tot_cnt_5']/login_scene_mdl['c_scene_tot_cnt_5']
login_scene_mdl['c_scene_h5_tot_fre_5'] = login_scene_mdl['c_scene_h5_tot_cnt_5']/login_scene_mdl['c_scene_tot_cnt_5']
login_scene_mdl['c_scene_android_tot_fre_5'] = login_scene_mdl['c_scene_android_tot_cnt_5']/login_scene_mdl['c_scene_tot_cnt_5']
login_scene_mdl['c_scene_ios_tot_fre_5'] = login_scene_mdl['c_scene_ios_tot_cnt_5']/login_scene_mdl['c_scene_tot_cnt_5']

#弄一个登陆，注册，，，，的总时长 =  平均时长*相对应次数
login_scene_mdl['c_scene_reg_sum_dur_10'] = login_scene_mdl['c_scene_reg_avg_dur_10']*login_scene_mdl['c_scene_reg_tot_cnt_10']
login_scene_mdl['c_scene_dl_sum_dur_10'] = login_scene_mdl['c_scene_dl_avg_dur_10']*login_scene_mdl['c_scene_dl_tot_cnt_10']
login_scene_mdl['c_scene_od_sum_dur_10'] = login_scene_mdl['c_scene_od_avg_dur_10']*login_scene_mdl['c_scene_od_tot_cnt_10']
login_scene_mdl['c_scene_rp_sum_dur_10'] = login_scene_mdl['c_scene_rp_avg_dur_10']*login_scene_mdl['c_scene_rp_tot_cnt_10']
login_scene_mdl['c_scene_xgxx_sum_dur_10'] = login_scene_mdl['c_scene_xgxx_avg_dur_10']*login_scene_mdl['c_scene_xgxx_tot_cnt_10']
login_scene_mdl['c_scene_plsp_sum_dur_10'] = login_scene_mdl['c_scene_plsp_avg_dur_10']*login_scene_mdl['c_scene_plsp_tot_cnt_10']
login_scene_mdl['c_scene_sczl_sum_dur_10'] = login_scene_mdl['c_scene_sczl_avg_dur_10']*login_scene_mdl['c_scene_sczl_tot_cnt_10']
login_scene_mdl['c_scene_sh_sum_dur_10'] = login_scene_mdl['c_scene_sh_avg_dur_10']*login_scene_mdl['c_scene_sh_tot_cnt_10']

login_scene_mdl['c_scene_reg_sum_dur_9'] = login_scene_mdl['c_scene_reg_avg_dur_9']*login_scene_mdl['c_scene_reg_tot_cnt_9']
login_scene_mdl['c_scene_dl_sum_dur_9'] = login_scene_mdl['c_scene_dl_avg_dur_9']*login_scene_mdl['c_scene_dl_tot_cnt_9']
login_scene_mdl['c_scene_od_sum_dur_9'] = login_scene_mdl['c_scene_od_avg_dur_9']*login_scene_mdl['c_scene_od_tot_cnt_9']
login_scene_mdl['c_scene_rp_sum_dur_9'] = login_scene_mdl['c_scene_rp_avg_dur_9']*login_scene_mdl['c_scene_rp_tot_cnt_9']
login_scene_mdl['c_scene_xgxx_sum_dur_9'] = login_scene_mdl['c_scene_xgxx_avg_dur_9']*login_scene_mdl['c_scene_xgxx_tot_cnt_9']
login_scene_mdl['c_scene_plsp_sum_dur_9'] = login_scene_mdl['c_scene_plsp_avg_dur_9']*login_scene_mdl['c_scene_plsp_tot_cnt_9']
login_scene_mdl['c_scene_sczl_sum_dur_9'] = login_scene_mdl['c_scene_sczl_avg_dur_9']*login_scene_mdl['c_scene_sczl_tot_cnt_9']
login_scene_mdl['c_scene_sh_sum_dur_9'] = login_scene_mdl['c_scene_sh_avg_dur_9']*login_scene_mdl['c_scene_sh_tot_cnt_9']

login_scene_mdl['c_scene_reg_sum_dur_x'] = login_scene_mdl['c_scene_reg_avg_dur_x']*login_scene_mdl['c_scene_reg_tot_cnt_x']
login_scene_mdl['c_scene_dl_sum_dur_x'] = login_scene_mdl['c_scene_dl_avg_dur_x']*login_scene_mdl['c_scene_dl_tot_cnt_x']
login_scene_mdl['c_scene_od_sum_dur_x'] = login_scene_mdl['c_scene_od_avg_dur_x']*login_scene_mdl['c_scene_od_tot_cnt_x']
login_scene_mdl['c_scene_rp_sum_dur_x'] = login_scene_mdl['c_scene_rp_avg_dur_x']*login_scene_mdl['c_scene_rp_tot_cnt_x']
login_scene_mdl['c_scene_xgxx_sum_dur_x'] = login_scene_mdl['c_scene_xgxx_avg_dur_x']*login_scene_mdl['c_scene_xgxx_tot_cnt_x']
login_scene_mdl['c_scene_plsp_sum_dur_x'] = login_scene_mdl['c_scene_plsp_avg_dur_x']*login_scene_mdl['c_scene_plsp_tot_cnt_x']
login_scene_mdl['c_scene_sczl_sum_dur_x'] = login_scene_mdl['c_scene_sczl_avg_dur_x']*login_scene_mdl['c_scene_sczl_tot_cnt_x']
login_scene_mdl['c_scene_sh_sum_dur_x'] = login_scene_mdl['c_scene_sh_avg_dur_x']*login_scene_mdl['c_scene_sh_tot_cnt_x']

login_scene_mdl['c_scene_reg_sum_dur_7'] = login_scene_mdl['c_scene_reg_avg_dur_7']*login_scene_mdl['c_scene_reg_tot_cnt_7']
login_scene_mdl['c_scene_dl_sum_dur_7'] = login_scene_mdl['c_scene_dl_avg_dur_7']*login_scene_mdl['c_scene_dl_tot_cnt_7']
login_scene_mdl['c_scene_od_sum_dur_7'] = login_scene_mdl['c_scene_od_avg_dur_7']*login_scene_mdl['c_scene_od_tot_cnt_7']
login_scene_mdl['c_scene_rp_sum_dur_7'] = login_scene_mdl['c_scene_rp_avg_dur_7']*login_scene_mdl['c_scene_rp_tot_cnt_7']
login_scene_mdl['c_scene_xgxx_sum_dur_7'] = login_scene_mdl['c_scene_xgxx_avg_dur_7']*login_scene_mdl['c_scene_xgxx_tot_cnt_7']
login_scene_mdl['c_scene_plsp_sum_dur_7'] = login_scene_mdl['c_scene_plsp_avg_dur_7']*login_scene_mdl['c_scene_plsp_tot_cnt_7']
login_scene_mdl['c_scene_sczl_sum_dur_7'] = login_scene_mdl['c_scene_sczl_avg_dur_7']*login_scene_mdl['c_scene_sczl_tot_cnt_7']
login_scene_mdl['c_scene_sh_sum_dur_7'] = login_scene_mdl['c_scene_sh_avg_dur_7']*login_scene_mdl['c_scene_sh_tot_cnt_7']

login_scene_mdl['c_scene_reg_sum_dur_6'] = login_scene_mdl['c_scene_reg_avg_dur_6']*login_scene_mdl['c_scene_reg_tot_cnt_6']
login_scene_mdl['c_scene_dl_sum_dur_6'] = login_scene_mdl['c_scene_dl_avg_dur_6']*login_scene_mdl['c_scene_dl_tot_cnt_6']
login_scene_mdl['c_scene_od_sum_dur_6'] = login_scene_mdl['c_scene_od_avg_dur_6']*login_scene_mdl['c_scene_od_tot_cnt_6']
login_scene_mdl['c_scene_rp_sum_dur_6'] = login_scene_mdl['c_scene_rp_avg_dur_6']*login_scene_mdl['c_scene_rp_tot_cnt_6']
login_scene_mdl['c_scene_xgxx_sum_dur_6'] = login_scene_mdl['c_scene_xgxx_avg_dur_6']*login_scene_mdl['c_scene_xgxx_tot_cnt_6']
login_scene_mdl['c_scene_plsp_sum_dur_6'] = login_scene_mdl['c_scene_plsp_avg_dur_6']*login_scene_mdl['c_scene_plsp_tot_cnt_6']
login_scene_mdl['c_scene_sczl_sum_dur_6'] = login_scene_mdl['c_scene_sczl_avg_dur_6']*login_scene_mdl['c_scene_sczl_tot_cnt_6']
login_scene_mdl['c_scene_sh_sum_dur_6'] = login_scene_mdl['c_scene_sh_avg_dur_6']*login_scene_mdl['c_scene_sh_tot_cnt_6']

login_scene_mdl['c_scene_reg_sum_dur_5'] = login_scene_mdl['c_scene_reg_avg_dur_5']*login_scene_mdl['c_scene_reg_tot_cnt_5']
login_scene_mdl['c_scene_dl_sum_dur_5'] = login_scene_mdl['c_scene_dl_avg_dur_5']*login_scene_mdl['c_scene_dl_tot_cnt_5']
login_scene_mdl['c_scene_od_sum_dur_5'] = login_scene_mdl['c_scene_od_avg_dur_5']*login_scene_mdl['c_scene_od_tot_cnt_5']
login_scene_mdl['c_scene_rp_sum_dur_5'] = login_scene_mdl['c_scene_rp_avg_dur_5']*login_scene_mdl['c_scene_rp_tot_cnt_5']
login_scene_mdl['c_scene_xgxx_sum_dur_5'] = login_scene_mdl['c_scene_xgxx_avg_dur_5']*login_scene_mdl['c_scene_xgxx_tot_cnt_5']
login_scene_mdl['c_scene_plsp_sum_dur_5'] = login_scene_mdl['c_scene_plsp_avg_dur_5']*login_scene_mdl['c_scene_plsp_tot_cnt_5']
login_scene_mdl['c_scene_sczl_sum_dur_5'] = login_scene_mdl['c_scene_sczl_avg_dur_5']*login_scene_mdl['c_scene_sczl_tot_cnt_5']
login_scene_mdl['c_scene_sh_sum_dur_5'] = login_scene_mdl['c_scene_sh_avg_dur_5']*login_scene_mdl['c_scene_sh_tot_cnt_5']

login_scene_mdl['c_scene_reg_sum_avg_dur_10'] = login_scene_mdl['c_scene_reg_sum_dur_10']/(login_scene_mdl['c_scene_tot_cnt_10']*login_scene_mdl['c_scene_log_avg_dur_10'])
login_scene_mdl['c_scene_dl_sum_avg_dur_10'] = login_scene_mdl['c_scene_dl_sum_dur_10']/(login_scene_mdl['c_scene_tot_cnt_10']*login_scene_mdl['c_scene_log_avg_dur_10'])
login_scene_mdl['c_scene_od_sum_avg_dur_10'] = login_scene_mdl['c_scene_od_sum_dur_10']/(login_scene_mdl['c_scene_tot_cnt_10']*login_scene_mdl['c_scene_log_avg_dur_10'])
login_scene_mdl['c_scene_rp_sum_avg_dur_10'] = login_scene_mdl['c_scene_rp_sum_dur_10']/(login_scene_mdl['c_scene_tot_cnt_10']*login_scene_mdl['c_scene_log_avg_dur_10'])
login_scene_mdl['c_scene_xgxx_sum_avg_dur_10'] = login_scene_mdl['c_scene_xgxx_sum_dur_10']/(login_scene_mdl['c_scene_tot_cnt_10']*login_scene_mdl['c_scene_log_avg_dur_10'])
login_scene_mdl['c_scene_plsp_sum_avg_dur_10'] = login_scene_mdl['c_scene_plsp_sum_dur_10']/(login_scene_mdl['c_scene_tot_cnt_10']*login_scene_mdl['c_scene_log_avg_dur_10'])
login_scene_mdl['c_scene_sczl_sum_avg_dur_10'] = login_scene_mdl['c_scene_sczl_sum_dur_10']/(login_scene_mdl['c_scene_tot_cnt_10']*login_scene_mdl['c_scene_log_avg_dur_10'])
login_scene_mdl['c_scene_sh_sum_avg_dur_10'] = login_scene_mdl['c_scene_sh_sum_dur_10']/(login_scene_mdl['c_scene_tot_cnt_10']*login_scene_mdl['c_scene_log_avg_dur_10'])

login_scene_mdl['c_scene_reg_sum_avg_dur_9'] = login_scene_mdl['c_scene_reg_sum_dur_9']/(login_scene_mdl['c_scene_tot_cnt_9']*login_scene_mdl['c_scene_log_avg_dur_9'])
login_scene_mdl['c_scene_dl_sum_avg_dur_9'] = login_scene_mdl['c_scene_dl_sum_dur_9']/(login_scene_mdl['c_scene_tot_cnt_9']*login_scene_mdl['c_scene_log_avg_dur_9'])
login_scene_mdl['c_scene_od_sum_avg_dur_9'] = login_scene_mdl['c_scene_od_sum_dur_9']/(login_scene_mdl['c_scene_tot_cnt_9']*login_scene_mdl['c_scene_log_avg_dur_9'])
login_scene_mdl['c_scene_rp_sum_avg_dur_9'] = login_scene_mdl['c_scene_rp_sum_dur_9']/(login_scene_mdl['c_scene_tot_cnt_9']*login_scene_mdl['c_scene_log_avg_dur_9'])
login_scene_mdl['c_scene_xgxx_sum_avg_dur_9'] = login_scene_mdl['c_scene_xgxx_sum_dur_9']/(login_scene_mdl['c_scene_tot_cnt_9']*login_scene_mdl['c_scene_log_avg_dur_9'])
login_scene_mdl['c_scene_plsp_sum_avg_dur_9'] = login_scene_mdl['c_scene_plsp_sum_dur_9']/(login_scene_mdl['c_scene_tot_cnt_9']*login_scene_mdl['c_scene_log_avg_dur_9'])
login_scene_mdl['c_scene_sczl_sum_avg_dur_9'] = login_scene_mdl['c_scene_sczl_sum_dur_9']/(login_scene_mdl['c_scene_tot_cnt_9']*login_scene_mdl['c_scene_log_avg_dur_9'])
login_scene_mdl['c_scene_sh_sum_avg_dur_9'] = login_scene_mdl['c_scene_sh_sum_dur_9']/(login_scene_mdl['c_scene_tot_cnt_9']*login_scene_mdl['c_scene_log_avg_dur_9'])

login_scene_mdl['c_scene_reg_sum_avg_dur_x'] = login_scene_mdl['c_scene_reg_sum_dur_x']/(login_scene_mdl['c_scene_tot_cnt_x']*login_scene_mdl['c_scene_log_avg_dur_x'])
login_scene_mdl['c_scene_dl_sum_avg_dur_x'] = login_scene_mdl['c_scene_dl_sum_dur_x']/(login_scene_mdl['c_scene_tot_cnt_x']*login_scene_mdl['c_scene_log_avg_dur_x'])
login_scene_mdl['c_scene_od_sum_avg_dur_x'] = login_scene_mdl['c_scene_od_sum_dur_x']/(login_scene_mdl['c_scene_tot_cnt_x']*login_scene_mdl['c_scene_log_avg_dur_x'])
login_scene_mdl['c_scene_rp_sum_avg_dur_x'] = login_scene_mdl['c_scene_rp_sum_dur_x']/(login_scene_mdl['c_scene_tot_cnt_x']*login_scene_mdl['c_scene_log_avg_dur_x'])
login_scene_mdl['c_scene_xgxx_sum_avg_dur_x'] = login_scene_mdl['c_scene_xgxx_sum_dur_x']/(login_scene_mdl['c_scene_tot_cnt_x']*login_scene_mdl['c_scene_log_avg_dur_x'])
login_scene_mdl['c_scene_plsp_sum_avg_dur_x'] = login_scene_mdl['c_scene_plsp_sum_dur_x']/(login_scene_mdl['c_scene_tot_cnt_x']*login_scene_mdl['c_scene_log_avg_dur_x'])
login_scene_mdl['c_scene_sczl_sum_avg_dur_x'] = login_scene_mdl['c_scene_sczl_sum_dur_x']/(login_scene_mdl['c_scene_tot_cnt_x']*login_scene_mdl['c_scene_log_avg_dur_x'])
login_scene_mdl['c_scene_sh_sum_avg_dur_x'] = login_scene_mdl['c_scene_sh_sum_dur_x']/(login_scene_mdl['c_scene_tot_cnt_x']*login_scene_mdl['c_scene_log_avg_dur_x'])

login_scene_mdl['c_scene_reg_sum_avg_dur_7'] = login_scene_mdl['c_scene_reg_sum_dur_7']/(login_scene_mdl['c_scene_tot_cnt_7']*login_scene_mdl['c_scene_log_avg_dur_7'])
login_scene_mdl['c_scene_dl_sum_avg_dur_7'] = login_scene_mdl['c_scene_dl_sum_dur_7']/(login_scene_mdl['c_scene_tot_cnt_7']*login_scene_mdl['c_scene_log_avg_dur_7'])
login_scene_mdl['c_scene_od_sum_avg_dur_7'] = login_scene_mdl['c_scene_od_sum_dur_7']/(login_scene_mdl['c_scene_tot_cnt_7']*login_scene_mdl['c_scene_log_avg_dur_7'])
login_scene_mdl['c_scene_rp_sum_avg_dur_7'] = login_scene_mdl['c_scene_rp_sum_dur_7']/(login_scene_mdl['c_scene_tot_cnt_7']*login_scene_mdl['c_scene_log_avg_dur_7'])
login_scene_mdl['c_scene_xgxx_sum_avg_dur_7'] = login_scene_mdl['c_scene_xgxx_sum_dur_7']/(login_scene_mdl['c_scene_tot_cnt_7']*login_scene_mdl['c_scene_log_avg_dur_7'])
login_scene_mdl['c_scene_plsp_sum_avg_dur_7'] = login_scene_mdl['c_scene_plsp_sum_dur_7']/(login_scene_mdl['c_scene_tot_cnt_7']*login_scene_mdl['c_scene_log_avg_dur_7'])
login_scene_mdl['c_scene_sczl_sum_avg_dur_7'] = login_scene_mdl['c_scene_sczl_sum_dur_7']/(login_scene_mdl['c_scene_tot_cnt_7']*login_scene_mdl['c_scene_log_avg_dur_7'])
login_scene_mdl['c_scene_sh_sum_avg_dur_7'] = login_scene_mdl['c_scene_sh_sum_dur_7']/(login_scene_mdl['c_scene_tot_cnt_7']*login_scene_mdl['c_scene_log_avg_dur_7'])

login_scene_mdl['c_scene_reg_sum_avg_dur_6'] = login_scene_mdl['c_scene_reg_sum_dur_6']/(login_scene_mdl['c_scene_tot_cnt_6']*login_scene_mdl['c_scene_log_avg_dur_6'])
login_scene_mdl['c_scene_dl_sum_avg_dur_6'] = login_scene_mdl['c_scene_dl_sum_dur_6']/(login_scene_mdl['c_scene_tot_cnt_6']*login_scene_mdl['c_scene_log_avg_dur_6'])
login_scene_mdl['c_scene_od_sum_avg_dur_6'] = login_scene_mdl['c_scene_od_sum_dur_6']/(login_scene_mdl['c_scene_tot_cnt_6']*login_scene_mdl['c_scene_log_avg_dur_6'])
login_scene_mdl['c_scene_rp_sum_avg_dur_6'] = login_scene_mdl['c_scene_rp_sum_dur_6']/(login_scene_mdl['c_scene_tot_cnt_6']*login_scene_mdl['c_scene_log_avg_dur_6'])
login_scene_mdl['c_scene_xgxx_sum_avg_dur_6'] = login_scene_mdl['c_scene_xgxx_sum_dur_6']/(login_scene_mdl['c_scene_tot_cnt_6']*login_scene_mdl['c_scene_log_avg_dur_6'])
login_scene_mdl['c_scene_plsp_sum_avg_dur_6'] = login_scene_mdl['c_scene_plsp_sum_dur_6']/(login_scene_mdl['c_scene_tot_cnt_6']*login_scene_mdl['c_scene_log_avg_dur_6'])
login_scene_mdl['c_scene_sczl_sum_avg_dur_6'] = login_scene_mdl['c_scene_sczl_sum_dur_6']/(login_scene_mdl['c_scene_tot_cnt_6']*login_scene_mdl['c_scene_log_avg_dur_6'])
login_scene_mdl['c_scene_sh_sum_avg_dur_6'] = login_scene_mdl['c_scene_sh_sum_dur_6']/(login_scene_mdl['c_scene_tot_cnt_6']*login_scene_mdl['c_scene_log_avg_dur_6'])

login_scene_mdl['c_scene_reg_sum_avg_dur_5'] = login_scene_mdl['c_scene_reg_sum_dur_5']/(login_scene_mdl['c_scene_tot_cnt_5']*login_scene_mdl['c_scene_log_avg_dur_5'])
login_scene_mdl['c_scene_dl_sum_avg_dur_5'] = login_scene_mdl['c_scene_dl_sum_dur_5']/(login_scene_mdl['c_scene_tot_cnt_5']*login_scene_mdl['c_scene_log_avg_dur_5'])
login_scene_mdl['c_scene_od_sum_avg_dur_5'] = login_scene_mdl['c_scene_od_sum_dur_5']/(login_scene_mdl['c_scene_tot_cnt_5']*login_scene_mdl['c_scene_log_avg_dur_5'])
login_scene_mdl['c_scene_rp_sum_avg_dur_5'] = login_scene_mdl['c_scene_rp_sum_dur_5']/(login_scene_mdl['c_scene_tot_cnt_5']*login_scene_mdl['c_scene_log_avg_dur_5'])
login_scene_mdl['c_scene_xgxx_sum_avg_dur_5'] = login_scene_mdl['c_scene_xgxx_sum_dur_5']/(login_scene_mdl['c_scene_tot_cnt_5']*login_scene_mdl['c_scene_log_avg_dur_5'])
login_scene_mdl['c_scene_plsp_sum_avg_dur_5'] = login_scene_mdl['c_scene_plsp_sum_dur_5']/(login_scene_mdl['c_scene_tot_cnt_5']*login_scene_mdl['c_scene_log_avg_dur_5'])
login_scene_mdl['c_scene_sczl_sum_avg_dur_5'] = login_scene_mdl['c_scene_sczl_sum_dur_5']/(login_scene_mdl['c_scene_tot_cnt_5']*login_scene_mdl['c_scene_log_avg_dur_5'])
login_scene_mdl['c_scene_sh_sum_avg_dur_5'] = login_scene_mdl['c_scene_sh_sum_dur_5']/(login_scene_mdl['c_scene_tot_cnt_5']*login_scene_mdl['c_scene_log_avg_dur_5'])

#去掉标准差小于0.1的数据
login_scene_mdl = login_scene_mdl.fillna(0)
login_scene_mdl_var = login_scene_mdl.var().reset_index()
login_scene_mdl_var.columns = ['index','val']
login_scene_mdl = login_scene_mdl.drop(login_scene_mdl_var[login_scene_mdl_var.val<0.1]['index'],axis=1)
login_scene_mdl.to_csv(r'features/login_scene_mdl.csv',index=False)


#测试集
login_scene_offtime = pd.read_csv(r"data\lexin_test\login_scene_offtime.csv")
import datetime as dt
login_scene_offtime['cyc_date'] = pd.to_datetime(login_scene_offtime['cyc_date'],format='%Y-%m-%d %H:%M:%S')
login_scene_offtime['cyc_date']=[i.month for i in login_scene_offtime["cyc_date"]]
login_scene_offtime = login_scene_offtime.drop('pyear_month',axis=1)

#应该先弄好每个人有的六行数据的每个特征的最大值，最小值，均值，方差，中位数，众值
login_scene_offtime_mean = login_scene_offtime.groupby(['fuid_md5']).agg(np.mean)
login_scene_offtime_max = login_scene_offtime.groupby(['fuid_md5']).agg(np.max)
login_scene_offtime_min = login_scene_offtime.groupby(['fuid_md5']).agg(np.min)
login_scene_offtime_std = login_scene_offtime.groupby(['fuid_md5']).agg(np.std)
login_scene_offtime_median = login_scene_offtime.groupby(['fuid_md5']).agg(np.median)

login_scene_offtime_mean = login_scene_offtime_mean.reset_index()
login_scene_offtime_max = login_scene_offtime_max.reset_index()
login_scene_offtime_min = login_scene_offtime_min.reset_index()
login_scene_offtime_std = login_scene_offtime_std.reset_index()
login_scene_offtime_median = login_scene_offtime_median.reset_index()

login_scene_offtime_mean = login_scene_offtime_mean.drop('cyc_date',axis=1)
login_scene_offtime_max = login_scene_offtime_max.drop('cyc_date',axis=1)
login_scene_offtime_min = login_scene_offtime_min.drop('cyc_date',axis=1)
login_scene_offtime_std = login_scene_offtime_std.drop('cyc_date',axis=1)
login_scene_offtime_median = login_scene_offtime_median.drop('cyc_date',axis=1)

#将均值，最大值，最小值，方差，中值拼接起来，应该是53*5+1列数据
login_scene_offtime_sta = pd.merge(login_scene_offtime_mean,login_scene_offtime_max,how='left',on ='fuid_md5',suffixes=['_mean', '_max'])
login_scene_offtime_sta = pd.merge(login_scene_offtime_sta,login_scene_offtime_min,how='left',on ='fuid_md5',suffixes=['', '_min'])
login_scene_offtime_sta = pd.merge(login_scene_offtime_sta,login_scene_offtime_std,how='left',on ='fuid_md5',suffixes=['', '_std'])
login_scene_offtime_sta = pd.merge(login_scene_offtime_sta,login_scene_offtime_median,how='left',on ='fuid_md5',suffixes=['', '_median'])

#获取7,8,9,10,11,12月份数据
login_scene_offtime_12 = login_scene_offtime[login_scene_offtime['cyc_date']==12]
login_scene_offtime_12 = login_scene_offtime_12.drop('cyc_date',axis=1)
login_scene_offtime_11 = login_scene_offtime[login_scene_offtime['cyc_date']==11]
login_scene_offtime_11 = login_scene_offtime_11.drop('cyc_date',axis=1)
login_scene_offtime_10 = login_scene_offtime[login_scene_offtime['cyc_date']==10]
login_scene_offtime_10 = login_scene_offtime_10.drop('cyc_date',axis=1)
login_scene_offtime_9 = login_scene_offtime[login_scene_offtime['cyc_date']==9]
login_scene_offtime_9 = login_scene_offtime_9.drop('cyc_date',axis=1)
login_scene_offtime_8 = login_scene_offtime[login_scene_offtime['cyc_date']==8]
login_scene_offtime_8 = login_scene_offtime_8.drop('cyc_date',axis=1)
login_scene_offtime_7 = login_scene_offtime[login_scene_offtime['cyc_date']==7]
login_scene_offtime_7 = login_scene_offtime_7.drop('cyc_date',axis=1)
login_scene_offtime_mon = pd.merge(login_scene_offtime_12,login_scene_offtime_11,how='left',on ='fuid_md5',suffixes=['_10', '_9'])
login_scene_offtime_mon = pd.merge(login_scene_offtime_mon,login_scene_offtime_10,how='left',on ='fuid_md5',suffixes=['', '_8'])
login_scene_offtime_mon = pd.merge(login_scene_offtime_mon,login_scene_offtime_9,how='left',on ='fuid_md5',suffixes=['', '_7'])
login_scene_offtime_mon = pd.merge(login_scene_offtime_mon,login_scene_offtime_8,how='left',on ='fuid_md5',suffixes=['', '_6'])
login_scene_offtime_mon = pd.merge(login_scene_offtime_mon,login_scene_offtime_7,how='left',on ='fuid_md5',suffixes=['', '_5'])
login_scene_offtime = pd.merge(login_scene_offtime_mon,login_scene_offtime_sta,how='left',on ='fuid_md5')

login_scene_offtime['c_scene_reg_tot_fre_10'] = login_scene_offtime['c_scene_reg_tot_cnt_10']/login_scene_offtime['c_scene_tot_cnt_10']
login_scene_offtime['c_scene_dl_tot_fre_10'] = login_scene_offtime['c_scene_dl_tot_cnt_10']/login_scene_offtime['c_scene_tot_cnt_10']
login_scene_offtime['c_scene_od_tot_fre_10'] = login_scene_offtime['c_scene_od_tot_cnt_10']/login_scene_offtime['c_scene_tot_cnt_10']
login_scene_offtime['c_scene_rp_tot_fre_10'] = login_scene_offtime['c_scene_rp_tot_cnt_10']/login_scene_offtime['c_scene_tot_cnt_10']
login_scene_offtime['c_scene_xgxx_tot_fre_10'] = login_scene_offtime['c_scene_xgxx_tot_cnt_10']/login_scene_offtime['c_scene_tot_cnt_10']
login_scene_offtime['c_scene_plsp_tot_fre_10'] = login_scene_offtime['c_scene_plsp_tot_cnt_10']/login_scene_offtime['c_scene_tot_cnt_10']
login_scene_offtime['c_scene_sczl_tot_fre_10'] = login_scene_offtime['c_scene_sczl_tot_cnt_10']/login_scene_offtime['c_scene_tot_cnt_10']
login_scene_offtime['c_scene_sh_tot_fre_10'] = login_scene_offtime['c_scene_sh_tot_cnt_10']/login_scene_offtime['c_scene_tot_cnt_10']

login_scene_offtime['c_scene_reg_tot_fre_9'] = login_scene_offtime['c_scene_reg_tot_cnt_9']/login_scene_offtime['c_scene_tot_cnt_9']
login_scene_offtime['c_scene_dl_tot_fre_9'] = login_scene_offtime['c_scene_dl_tot_cnt_9']/login_scene_offtime['c_scene_tot_cnt_9']
login_scene_offtime['c_scene_od_tot_fre_9'] = login_scene_offtime['c_scene_od_tot_cnt_9']/login_scene_offtime['c_scene_tot_cnt_9']
login_scene_offtime['c_scene_rp_tot_fre_9'] = login_scene_offtime['c_scene_rp_tot_cnt_9']/login_scene_offtime['c_scene_tot_cnt_9']
login_scene_offtime['c_scene_xgxx_tot_fre_9'] = login_scene_offtime['c_scene_xgxx_tot_cnt_9']/login_scene_offtime['c_scene_tot_cnt_9']
login_scene_offtime['c_scene_plsp_tot_fre_9'] = login_scene_offtime['c_scene_plsp_tot_cnt_9']/login_scene_offtime['c_scene_tot_cnt_9']
login_scene_offtime['c_scene_sczl_tot_fre_9'] = login_scene_offtime['c_scene_sczl_tot_cnt_9']/login_scene_offtime['c_scene_tot_cnt_9']
login_scene_offtime['c_scene_sh_tot_fre_9'] = login_scene_offtime['c_scene_sh_tot_cnt_9']/login_scene_offtime['c_scene_tot_cnt_9']

login_scene_offtime['c_scene_reg_tot_fre_x'] = login_scene_offtime['c_scene_reg_tot_cnt_x']/login_scene_offtime['c_scene_tot_cnt_x']
login_scene_offtime['c_scene_dl_tot_fre_x'] = login_scene_offtime['c_scene_dl_tot_cnt_x']/login_scene_offtime['c_scene_tot_cnt_x']
login_scene_offtime['c_scene_od_tot_fre_x'] = login_scene_offtime['c_scene_od_tot_cnt_x']/login_scene_offtime['c_scene_tot_cnt_x']
login_scene_offtime['c_scene_rp_tot_fre_x'] = login_scene_offtime['c_scene_rp_tot_cnt_x']/login_scene_offtime['c_scene_tot_cnt_x']
login_scene_offtime['c_scene_xgxx_tot_fre_x'] = login_scene_offtime['c_scene_xgxx_tot_cnt_x']/login_scene_offtime['c_scene_tot_cnt_x']
login_scene_offtime['c_scene_plsp_tot_fre_x'] = login_scene_offtime['c_scene_plsp_tot_cnt_x']/login_scene_offtime['c_scene_tot_cnt_x']
login_scene_offtime['c_scene_sczl_tot_fre_x'] = login_scene_offtime['c_scene_sczl_tot_cnt_x']/login_scene_offtime['c_scene_tot_cnt_x']
login_scene_offtime['c_scene_sh_tot_fre_x'] = login_scene_offtime['c_scene_sh_tot_cnt_x']/login_scene_offtime['c_scene_tot_cnt_x']

login_scene_offtime['c_scene_reg_tot_fre_7'] = login_scene_offtime['c_scene_reg_tot_cnt_7']/login_scene_offtime['c_scene_tot_cnt_7']
login_scene_offtime['c_scene_dl_tot_fre_7'] = login_scene_offtime['c_scene_dl_tot_cnt_7']/login_scene_offtime['c_scene_tot_cnt_7']
login_scene_offtime['c_scene_od_tot_fre_7'] = login_scene_offtime['c_scene_od_tot_cnt_7']/login_scene_offtime['c_scene_tot_cnt_7']
login_scene_offtime['c_scene_rp_tot_fre_7'] = login_scene_offtime['c_scene_rp_tot_cnt_7']/login_scene_offtime['c_scene_tot_cnt_7']
login_scene_offtime['c_scene_xgxx_tot_fre_7'] = login_scene_offtime['c_scene_xgxx_tot_cnt_7']/login_scene_offtime['c_scene_tot_cnt_7']
login_scene_offtime['c_scene_plsp_tot_fre_7'] = login_scene_offtime['c_scene_plsp_tot_cnt_7']/login_scene_offtime['c_scene_tot_cnt_7']
login_scene_offtime['c_scene_sczl_tot_fre_7'] = login_scene_offtime['c_scene_sczl_tot_cnt_7']/login_scene_offtime['c_scene_tot_cnt_7']
login_scene_offtime['c_scene_sh_tot_fre_7'] = login_scene_offtime['c_scene_sh_tot_cnt_7']/login_scene_offtime['c_scene_tot_cnt_7']

login_scene_offtime['c_scene_reg_tot_fre_6'] = login_scene_offtime['c_scene_reg_tot_cnt_6']/login_scene_offtime['c_scene_tot_cnt_6']
login_scene_offtime['c_scene_dl_tot_fre_6'] = login_scene_offtime['c_scene_dl_tot_cnt_6']/login_scene_offtime['c_scene_tot_cnt_6']
login_scene_offtime['c_scene_od_tot_fre_6'] = login_scene_offtime['c_scene_od_tot_cnt_6']/login_scene_offtime['c_scene_tot_cnt_6']
login_scene_offtime['c_scene_rp_tot_fre_6'] = login_scene_offtime['c_scene_rp_tot_cnt_6']/login_scene_offtime['c_scene_tot_cnt_6']
login_scene_offtime['c_scene_xgxx_tot_fre_6'] = login_scene_offtime['c_scene_xgxx_tot_cnt_6']/login_scene_offtime['c_scene_tot_cnt_6']
login_scene_offtime['c_scene_plsp_tot_fre_6'] = login_scene_offtime['c_scene_plsp_tot_cnt_6']/login_scene_offtime['c_scene_tot_cnt_6']
login_scene_offtime['c_scene_sczl_tot_fre_6'] = login_scene_offtime['c_scene_sczl_tot_cnt_6']/login_scene_offtime['c_scene_tot_cnt_6']
login_scene_offtime['c_scene_sh_tot_fre_6'] = login_scene_offtime['c_scene_sh_tot_cnt_6']/login_scene_offtime['c_scene_tot_cnt_6']

login_scene_offtime['c_scene_reg_tot_fre_5'] = login_scene_offtime['c_scene_reg_tot_cnt_5']/login_scene_offtime['c_scene_tot_cnt_5']
login_scene_offtime['c_scene_dl_tot_fre_5'] = login_scene_offtime['c_scene_dl_tot_cnt_5']/login_scene_offtime['c_scene_tot_cnt_5']
login_scene_offtime['c_scene_od_tot_fre_5'] = login_scene_offtime['c_scene_od_tot_cnt_5']/login_scene_offtime['c_scene_tot_cnt_5']
login_scene_offtime['c_scene_rp_tot_fre_5'] = login_scene_offtime['c_scene_rp_tot_cnt_5']/login_scene_offtime['c_scene_tot_cnt_5']
login_scene_offtime['c_scene_xgxx_tot_fre_5'] = login_scene_offtime['c_scene_xgxx_tot_cnt_5']/login_scene_offtime['c_scene_tot_cnt_5']
login_scene_offtime['c_scene_plsp_tot_fre_5'] = login_scene_offtime['c_scene_plsp_tot_cnt_5']/login_scene_offtime['c_scene_tot_cnt_5']
login_scene_offtime['c_scene_sczl_tot_fre_5'] = login_scene_offtime['c_scene_sczl_tot_cnt_5']/login_scene_offtime['c_scene_tot_cnt_5']
login_scene_offtime['c_scene_sh_tot_fre_5'] = login_scene_offtime['c_scene_sh_tot_cnt_5']/login_scene_offtime['c_scene_tot_cnt_5']

login_scene_offtime['c_scene_pc_tot_fre_10'] = login_scene_offtime['c_scene_pc_tot_cnt_10']/login_scene_offtime['c_scene_tot_cnt_10']
login_scene_offtime['c_scene_app_tot_fre_10'] = login_scene_offtime['c_scene_app_tot_cnt_10']/login_scene_offtime['c_scene_tot_cnt_10']
login_scene_offtime['c_scene_h5_tot_fre_10'] = login_scene_offtime['c_scene_h5_tot_cnt_10']/login_scene_offtime['c_scene_tot_cnt_10']
login_scene_offtime['c_scene_android_tot_fre_10'] = login_scene_offtime['c_scene_android_tot_cnt_10']/login_scene_offtime['c_scene_tot_cnt_10']
login_scene_offtime['c_scene_ios_tot_fre_10'] = login_scene_offtime['c_scene_ios_tot_cnt_10']/login_scene_offtime['c_scene_tot_cnt_10']

login_scene_offtime['c_scene_pc_tot_fre_9'] = login_scene_offtime['c_scene_pc_tot_cnt_9']/login_scene_offtime['c_scene_tot_cnt_9']
login_scene_offtime['c_scene_app_tot_fre_9'] = login_scene_offtime['c_scene_app_tot_cnt_9']/login_scene_offtime['c_scene_tot_cnt_9']
login_scene_offtime['c_scene_h5_tot_fre_9'] = login_scene_offtime['c_scene_h5_tot_cnt_9']/login_scene_offtime['c_scene_tot_cnt_9']
login_scene_offtime['c_scene_android_tot_fre_9'] = login_scene_offtime['c_scene_android_tot_cnt_9']/login_scene_offtime['c_scene_tot_cnt_9']
login_scene_offtime['c_scene_ios_tot_fre_9'] = login_scene_offtime['c_scene_ios_tot_cnt_9']/login_scene_offtime['c_scene_tot_cnt_9']

login_scene_offtime['c_scene_pc_tot_fre_x'] = login_scene_offtime['c_scene_pc_tot_cnt_x']/login_scene_offtime['c_scene_tot_cnt_x']
login_scene_offtime['c_scene_app_tot_fre_x'] = login_scene_offtime['c_scene_app_tot_cnt_x']/login_scene_offtime['c_scene_tot_cnt_x']
login_scene_offtime['c_scene_h5_tot_fre_x'] = login_scene_offtime['c_scene_h5_tot_cnt_x']/login_scene_offtime['c_scene_tot_cnt_x']
login_scene_offtime['c_scene_android_tot_fre_x'] = login_scene_offtime['c_scene_android_tot_cnt_x']/login_scene_offtime['c_scene_tot_cnt_x']
login_scene_offtime['c_scene_ios_tot_fre_x'] = login_scene_offtime['c_scene_ios_tot_cnt_x']/login_scene_offtime['c_scene_tot_cnt_x']

login_scene_offtime['c_scene_pc_tot_fre_7'] = login_scene_offtime['c_scene_pc_tot_cnt_7']/login_scene_offtime['c_scene_tot_cnt_7']
login_scene_offtime['c_scene_app_tot_fre_7'] = login_scene_offtime['c_scene_app_tot_cnt_7']/login_scene_offtime['c_scene_tot_cnt_7']
login_scene_offtime['c_scene_h5_tot_fre_7'] = login_scene_offtime['c_scene_h5_tot_cnt_7']/login_scene_offtime['c_scene_tot_cnt_7']
login_scene_offtime['c_scene_android_tot_fre_7'] = login_scene_offtime['c_scene_android_tot_cnt_7']/login_scene_offtime['c_scene_tot_cnt_7']
login_scene_offtime['c_scene_ios_tot_fre_7'] = login_scene_offtime['c_scene_ios_tot_cnt_7']/login_scene_offtime['c_scene_tot_cnt_7']

login_scene_offtime['c_scene_pc_tot_fre_6'] = login_scene_offtime['c_scene_pc_tot_cnt_6']/login_scene_offtime['c_scene_tot_cnt_6']
login_scene_offtime['c_scene_app_tot_fre_6'] = login_scene_offtime['c_scene_app_tot_cnt_6']/login_scene_offtime['c_scene_tot_cnt_6']
login_scene_offtime['c_scene_h5_tot_fre_6'] = login_scene_offtime['c_scene_h5_tot_cnt_6']/login_scene_offtime['c_scene_tot_cnt_6']
login_scene_offtime['c_scene_android_tot_fre_6'] = login_scene_offtime['c_scene_android_tot_cnt_6']/login_scene_offtime['c_scene_tot_cnt_6']
login_scene_offtime['c_scene_ios_tot_fre_6'] = login_scene_offtime['c_scene_ios_tot_cnt_6']/login_scene_offtime['c_scene_tot_cnt_6']

login_scene_offtime['c_scene_pc_tot_fre_5'] = login_scene_offtime['c_scene_pc_tot_cnt_5']/login_scene_offtime['c_scene_tot_cnt_5']
login_scene_offtime['c_scene_app_tot_fre_5'] = login_scene_offtime['c_scene_app_tot_cnt_5']/login_scene_offtime['c_scene_tot_cnt_5']
login_scene_offtime['c_scene_h5_tot_fre_5'] = login_scene_offtime['c_scene_h5_tot_cnt_5']/login_scene_offtime['c_scene_tot_cnt_5']
login_scene_offtime['c_scene_android_tot_fre_5'] = login_scene_offtime['c_scene_android_tot_cnt_5']/login_scene_offtime['c_scene_tot_cnt_5']
login_scene_offtime['c_scene_ios_tot_fre_5'] = login_scene_offtime['c_scene_ios_tot_cnt_5']/login_scene_offtime['c_scene_tot_cnt_5']

login_scene_offtime['c_scene_reg_sum_dur_10'] = login_scene_offtime['c_scene_reg_avg_dur_10']*login_scene_offtime['c_scene_reg_tot_cnt_10']
login_scene_offtime['c_scene_dl_sum_dur_10'] = login_scene_offtime['c_scene_dl_avg_dur_10']*login_scene_offtime['c_scene_dl_tot_cnt_10']
login_scene_offtime['c_scene_od_sum_dur_10'] = login_scene_offtime['c_scene_od_avg_dur_10']*login_scene_offtime['c_scene_od_tot_cnt_10']
login_scene_offtime['c_scene_rp_sum_dur_10'] = login_scene_offtime['c_scene_rp_avg_dur_10']*login_scene_offtime['c_scene_rp_tot_cnt_10']
login_scene_offtime['c_scene_xgxx_sum_dur_10'] = login_scene_offtime['c_scene_xgxx_avg_dur_10']*login_scene_offtime['c_scene_xgxx_tot_cnt_10']
login_scene_offtime['c_scene_plsp_sum_dur_10'] = login_scene_offtime['c_scene_plsp_avg_dur_10']*login_scene_offtime['c_scene_plsp_tot_cnt_10']
login_scene_offtime['c_scene_sczl_sum_dur_10'] = login_scene_offtime['c_scene_sczl_avg_dur_10']*login_scene_offtime['c_scene_sczl_tot_cnt_10']
login_scene_offtime['c_scene_sh_sum_dur_10'] = login_scene_offtime['c_scene_sh_avg_dur_10']*login_scene_offtime['c_scene_sh_tot_cnt_10']

login_scene_offtime['c_scene_reg_sum_dur_9'] = login_scene_offtime['c_scene_reg_avg_dur_9']*login_scene_offtime['c_scene_reg_tot_cnt_9']
login_scene_offtime['c_scene_dl_sum_dur_9'] = login_scene_offtime['c_scene_dl_avg_dur_9']*login_scene_offtime['c_scene_dl_tot_cnt_9']
login_scene_offtime['c_scene_od_sum_dur_9'] = login_scene_offtime['c_scene_od_avg_dur_9']*login_scene_offtime['c_scene_od_tot_cnt_9']
login_scene_offtime['c_scene_rp_sum_dur_9'] = login_scene_offtime['c_scene_rp_avg_dur_9']*login_scene_offtime['c_scene_rp_tot_cnt_9']
login_scene_offtime['c_scene_xgxx_sum_dur_9'] = login_scene_offtime['c_scene_xgxx_avg_dur_9']*login_scene_offtime['c_scene_xgxx_tot_cnt_9']
login_scene_offtime['c_scene_plsp_sum_dur_9'] = login_scene_offtime['c_scene_plsp_avg_dur_9']*login_scene_offtime['c_scene_plsp_tot_cnt_9']
login_scene_offtime['c_scene_sczl_sum_dur_9'] = login_scene_offtime['c_scene_sczl_avg_dur_9']*login_scene_offtime['c_scene_sczl_tot_cnt_9']
login_scene_offtime['c_scene_sh_sum_dur_9'] = login_scene_offtime['c_scene_sh_avg_dur_9']*login_scene_offtime['c_scene_sh_tot_cnt_9']

login_scene_offtime['c_scene_reg_sum_dur_x'] = login_scene_offtime['c_scene_reg_avg_dur_x']*login_scene_offtime['c_scene_reg_tot_cnt_x']
login_scene_offtime['c_scene_dl_sum_dur_x'] = login_scene_offtime['c_scene_dl_avg_dur_x']*login_scene_offtime['c_scene_dl_tot_cnt_x']
login_scene_offtime['c_scene_od_sum_dur_x'] = login_scene_offtime['c_scene_od_avg_dur_x']*login_scene_offtime['c_scene_od_tot_cnt_x']
login_scene_offtime['c_scene_rp_sum_dur_x'] = login_scene_offtime['c_scene_rp_avg_dur_x']*login_scene_offtime['c_scene_rp_tot_cnt_x']
login_scene_offtime['c_scene_xgxx_sum_dur_x'] = login_scene_offtime['c_scene_xgxx_avg_dur_x']*login_scene_offtime['c_scene_xgxx_tot_cnt_x']
login_scene_offtime['c_scene_plsp_sum_dur_x'] = login_scene_offtime['c_scene_plsp_avg_dur_x']*login_scene_offtime['c_scene_plsp_tot_cnt_x']
login_scene_offtime['c_scene_sczl_sum_dur_x'] = login_scene_offtime['c_scene_sczl_avg_dur_x']*login_scene_offtime['c_scene_sczl_tot_cnt_x']
login_scene_offtime['c_scene_sh_sum_dur_x'] = login_scene_offtime['c_scene_sh_avg_dur_x']*login_scene_offtime['c_scene_sh_tot_cnt_x']

login_scene_offtime['c_scene_reg_sum_dur_7'] = login_scene_offtime['c_scene_reg_avg_dur_7']*login_scene_offtime['c_scene_reg_tot_cnt_7']
login_scene_offtime['c_scene_dl_sum_dur_7'] = login_scene_offtime['c_scene_dl_avg_dur_7']*login_scene_offtime['c_scene_dl_tot_cnt_7']
login_scene_offtime['c_scene_od_sum_dur_7'] = login_scene_offtime['c_scene_od_avg_dur_7']*login_scene_offtime['c_scene_od_tot_cnt_7']
login_scene_offtime['c_scene_rp_sum_dur_7'] = login_scene_offtime['c_scene_rp_avg_dur_7']*login_scene_offtime['c_scene_rp_tot_cnt_7']
login_scene_offtime['c_scene_xgxx_sum_dur_7'] = login_scene_offtime['c_scene_xgxx_avg_dur_7']*login_scene_offtime['c_scene_xgxx_tot_cnt_7']
login_scene_offtime['c_scene_plsp_sum_dur_7'] = login_scene_offtime['c_scene_plsp_avg_dur_7']*login_scene_offtime['c_scene_plsp_tot_cnt_7']
login_scene_offtime['c_scene_sczl_sum_dur_7'] = login_scene_offtime['c_scene_sczl_avg_dur_7']*login_scene_offtime['c_scene_sczl_tot_cnt_7']
login_scene_offtime['c_scene_sh_sum_dur_7'] = login_scene_offtime['c_scene_sh_avg_dur_7']*login_scene_offtime['c_scene_sh_tot_cnt_7']

login_scene_offtime['c_scene_reg_sum_dur_6'] = login_scene_offtime['c_scene_reg_avg_dur_6']*login_scene_offtime['c_scene_reg_tot_cnt_6']
login_scene_offtime['c_scene_dl_sum_dur_6'] = login_scene_offtime['c_scene_dl_avg_dur_6']*login_scene_offtime['c_scene_dl_tot_cnt_6']
login_scene_offtime['c_scene_od_sum_dur_6'] = login_scene_offtime['c_scene_od_avg_dur_6']*login_scene_offtime['c_scene_od_tot_cnt_6']
login_scene_offtime['c_scene_rp_sum_dur_6'] = login_scene_offtime['c_scene_rp_avg_dur_6']*login_scene_offtime['c_scene_rp_tot_cnt_6']
login_scene_offtime['c_scene_xgxx_sum_dur_6'] = login_scene_offtime['c_scene_xgxx_avg_dur_6']*login_scene_offtime['c_scene_xgxx_tot_cnt_6']
login_scene_offtime['c_scene_plsp_sum_dur_6'] = login_scene_offtime['c_scene_plsp_avg_dur_6']*login_scene_offtime['c_scene_plsp_tot_cnt_6']
login_scene_offtime['c_scene_sczl_sum_dur_6'] = login_scene_offtime['c_scene_sczl_avg_dur_6']*login_scene_offtime['c_scene_sczl_tot_cnt_6']
login_scene_offtime['c_scene_sh_sum_dur_6'] = login_scene_offtime['c_scene_sh_avg_dur_6']*login_scene_offtime['c_scene_sh_tot_cnt_6']

login_scene_offtime['c_scene_reg_sum_dur_5'] = login_scene_offtime['c_scene_reg_avg_dur_5']*login_scene_offtime['c_scene_reg_tot_cnt_5']
login_scene_offtime['c_scene_dl_sum_dur_5'] = login_scene_offtime['c_scene_dl_avg_dur_5']*login_scene_offtime['c_scene_dl_tot_cnt_5']
login_scene_offtime['c_scene_od_sum_dur_5'] = login_scene_offtime['c_scene_od_avg_dur_5']*login_scene_offtime['c_scene_od_tot_cnt_5']
login_scene_offtime['c_scene_rp_sum_dur_5'] = login_scene_offtime['c_scene_rp_avg_dur_5']*login_scene_offtime['c_scene_rp_tot_cnt_5']
login_scene_offtime['c_scene_xgxx_sum_dur_5'] = login_scene_offtime['c_scene_xgxx_avg_dur_5']*login_scene_offtime['c_scene_xgxx_tot_cnt_5']
login_scene_offtime['c_scene_plsp_sum_dur_5'] = login_scene_offtime['c_scene_plsp_avg_dur_5']*login_scene_offtime['c_scene_plsp_tot_cnt_5']
login_scene_offtime['c_scene_sczl_sum_dur_5'] = login_scene_offtime['c_scene_sczl_avg_dur_5']*login_scene_offtime['c_scene_sczl_tot_cnt_5']
login_scene_offtime['c_scene_sh_sum_dur_5'] = login_scene_offtime['c_scene_sh_avg_dur_5']*login_scene_offtime['c_scene_sh_tot_cnt_5']

login_scene_offtime['c_scene_reg_sum_avg_dur_10'] = login_scene_offtime['c_scene_reg_sum_dur_10']/(login_scene_offtime['c_scene_tot_cnt_10']*login_scene_offtime['c_scene_log_avg_dur_10'])
login_scene_offtime['c_scene_dl_sum_avg_dur_10'] = login_scene_offtime['c_scene_dl_sum_dur_10']/(login_scene_offtime['c_scene_tot_cnt_10']*login_scene_offtime['c_scene_log_avg_dur_10'])
login_scene_offtime['c_scene_od_sum_avg_dur_10'] = login_scene_offtime['c_scene_od_sum_dur_10']/(login_scene_offtime['c_scene_tot_cnt_10']*login_scene_offtime['c_scene_log_avg_dur_10'])
login_scene_offtime['c_scene_rp_sum_avg_dur_10'] = login_scene_offtime['c_scene_rp_sum_dur_10']/(login_scene_offtime['c_scene_tot_cnt_10']*login_scene_offtime['c_scene_log_avg_dur_10'])
login_scene_offtime['c_scene_xgxx_sum_avg_dur_10'] = login_scene_offtime['c_scene_xgxx_sum_dur_10']/(login_scene_offtime['c_scene_tot_cnt_10']*login_scene_offtime['c_scene_log_avg_dur_10'])
login_scene_offtime['c_scene_plsp_sum_avg_dur_10'] = login_scene_offtime['c_scene_plsp_sum_dur_10']/(login_scene_offtime['c_scene_tot_cnt_10']*login_scene_offtime['c_scene_log_avg_dur_10'])
login_scene_offtime['c_scene_sczl_sum_avg_dur_10'] = login_scene_offtime['c_scene_sczl_sum_dur_10']/(login_scene_offtime['c_scene_tot_cnt_10']*login_scene_offtime['c_scene_log_avg_dur_10'])
login_scene_offtime['c_scene_sh_sum_avg_dur_10'] = login_scene_offtime['c_scene_sh_sum_dur_10']/(login_scene_offtime['c_scene_tot_cnt_10']*login_scene_offtime['c_scene_log_avg_dur_10'])

login_scene_offtime['c_scene_reg_sum_avg_dur_9'] = login_scene_offtime['c_scene_reg_sum_dur_9']/(login_scene_offtime['c_scene_tot_cnt_9']*login_scene_offtime['c_scene_log_avg_dur_9'])
login_scene_offtime['c_scene_dl_sum_avg_dur_9'] = login_scene_offtime['c_scene_dl_sum_dur_9']/(login_scene_offtime['c_scene_tot_cnt_9']*login_scene_offtime['c_scene_log_avg_dur_9'])
login_scene_offtime['c_scene_od_sum_avg_dur_9'] = login_scene_offtime['c_scene_od_sum_dur_9']/(login_scene_offtime['c_scene_tot_cnt_9']*login_scene_offtime['c_scene_log_avg_dur_9'])
login_scene_offtime['c_scene_rp_sum_avg_dur_9'] = login_scene_offtime['c_scene_rp_sum_dur_9']/(login_scene_offtime['c_scene_tot_cnt_9']*login_scene_offtime['c_scene_log_avg_dur_9'])
login_scene_offtime['c_scene_xgxx_sum_avg_dur_9'] = login_scene_offtime['c_scene_xgxx_sum_dur_9']/(login_scene_offtime['c_scene_tot_cnt_9']*login_scene_offtime['c_scene_log_avg_dur_9'])
login_scene_offtime['c_scene_plsp_sum_avg_dur_9'] = login_scene_offtime['c_scene_plsp_sum_dur_9']/(login_scene_offtime['c_scene_tot_cnt_9']*login_scene_offtime['c_scene_log_avg_dur_9'])
login_scene_offtime['c_scene_sczl_sum_avg_dur_9'] = login_scene_offtime['c_scene_sczl_sum_dur_9']/(login_scene_offtime['c_scene_tot_cnt_9']*login_scene_offtime['c_scene_log_avg_dur_9'])
login_scene_offtime['c_scene_sh_sum_avg_dur_9'] = login_scene_offtime['c_scene_sh_sum_dur_9']/(login_scene_offtime['c_scene_tot_cnt_9']*login_scene_offtime['c_scene_log_avg_dur_9'])

login_scene_offtime['c_scene_reg_sum_avg_dur_x'] = login_scene_offtime['c_scene_reg_sum_dur_x']/(login_scene_offtime['c_scene_tot_cnt_x']*login_scene_offtime['c_scene_log_avg_dur_x'])
login_scene_offtime['c_scene_dl_sum_avg_dur_x'] = login_scene_offtime['c_scene_dl_sum_dur_x']/(login_scene_offtime['c_scene_tot_cnt_x']*login_scene_offtime['c_scene_log_avg_dur_x'])
login_scene_offtime['c_scene_od_sum_avg_dur_x'] = login_scene_offtime['c_scene_od_sum_dur_x']/(login_scene_offtime['c_scene_tot_cnt_x']*login_scene_offtime['c_scene_log_avg_dur_x'])
login_scene_offtime['c_scene_rp_sum_avg_dur_x'] = login_scene_offtime['c_scene_rp_sum_dur_x']/(login_scene_offtime['c_scene_tot_cnt_x']*login_scene_offtime['c_scene_log_avg_dur_x'])
login_scene_offtime['c_scene_xgxx_sum_avg_dur_x'] = login_scene_offtime['c_scene_xgxx_sum_dur_x']/(login_scene_offtime['c_scene_tot_cnt_x']*login_scene_offtime['c_scene_log_avg_dur_x'])
login_scene_offtime['c_scene_plsp_sum_avg_dur_x'] = login_scene_offtime['c_scene_plsp_sum_dur_x']/(login_scene_offtime['c_scene_tot_cnt_x']*login_scene_offtime['c_scene_log_avg_dur_x'])
login_scene_offtime['c_scene_sczl_sum_avg_dur_x'] = login_scene_offtime['c_scene_sczl_sum_dur_x']/(login_scene_offtime['c_scene_tot_cnt_x']*login_scene_offtime['c_scene_log_avg_dur_x'])
login_scene_offtime['c_scene_sh_sum_avg_dur_x'] = login_scene_offtime['c_scene_sh_sum_dur_x']/(login_scene_offtime['c_scene_tot_cnt_x']*login_scene_offtime['c_scene_log_avg_dur_x'])

login_scene_offtime['c_scene_reg_sum_avg_dur_7'] = login_scene_offtime['c_scene_reg_sum_dur_7']/(login_scene_offtime['c_scene_tot_cnt_7']*login_scene_offtime['c_scene_log_avg_dur_7'])
login_scene_offtime['c_scene_dl_sum_avg_dur_7'] = login_scene_offtime['c_scene_dl_sum_dur_7']/(login_scene_offtime['c_scene_tot_cnt_7']*login_scene_offtime['c_scene_log_avg_dur_7'])
login_scene_offtime['c_scene_od_sum_avg_dur_7'] = login_scene_offtime['c_scene_od_sum_dur_7']/(login_scene_offtime['c_scene_tot_cnt_7']*login_scene_offtime['c_scene_log_avg_dur_7'])
login_scene_offtime['c_scene_rp_sum_avg_dur_7'] = login_scene_offtime['c_scene_rp_sum_dur_7']/(login_scene_offtime['c_scene_tot_cnt_7']*login_scene_offtime['c_scene_log_avg_dur_7'])
login_scene_offtime['c_scene_xgxx_sum_avg_dur_7'] = login_scene_offtime['c_scene_xgxx_sum_dur_7']/(login_scene_offtime['c_scene_tot_cnt_7']*login_scene_offtime['c_scene_log_avg_dur_7'])
login_scene_offtime['c_scene_plsp_sum_avg_dur_7'] = login_scene_offtime['c_scene_plsp_sum_dur_7']/(login_scene_offtime['c_scene_tot_cnt_7']*login_scene_offtime['c_scene_log_avg_dur_7'])
login_scene_offtime['c_scene_sczl_sum_avg_dur_7'] = login_scene_offtime['c_scene_sczl_sum_dur_7']/(login_scene_offtime['c_scene_tot_cnt_7']*login_scene_offtime['c_scene_log_avg_dur_7'])
login_scene_offtime['c_scene_sh_sum_avg_dur_7'] = login_scene_offtime['c_scene_sh_sum_dur_7']/(login_scene_offtime['c_scene_tot_cnt_7']*login_scene_offtime['c_scene_log_avg_dur_7'])

login_scene_offtime['c_scene_reg_sum_avg_dur_6'] = login_scene_offtime['c_scene_reg_sum_dur_6']/(login_scene_offtime['c_scene_tot_cnt_6']*login_scene_offtime['c_scene_log_avg_dur_6'])
login_scene_offtime['c_scene_dl_sum_avg_dur_6'] = login_scene_offtime['c_scene_dl_sum_dur_6']/(login_scene_offtime['c_scene_tot_cnt_6']*login_scene_offtime['c_scene_log_avg_dur_6'])
login_scene_offtime['c_scene_od_sum_avg_dur_6'] = login_scene_offtime['c_scene_od_sum_dur_6']/(login_scene_offtime['c_scene_tot_cnt_6']*login_scene_offtime['c_scene_log_avg_dur_6'])
login_scene_offtime['c_scene_rp_sum_avg_dur_6'] = login_scene_offtime['c_scene_rp_sum_dur_6']/(login_scene_offtime['c_scene_tot_cnt_6']*login_scene_offtime['c_scene_log_avg_dur_6'])
login_scene_offtime['c_scene_xgxx_sum_avg_dur_6'] = login_scene_offtime['c_scene_xgxx_sum_dur_6']/(login_scene_offtime['c_scene_tot_cnt_6']*login_scene_offtime['c_scene_log_avg_dur_6'])
login_scene_offtime['c_scene_plsp_sum_avg_dur_6'] = login_scene_offtime['c_scene_plsp_sum_dur_6']/(login_scene_offtime['c_scene_tot_cnt_6']*login_scene_offtime['c_scene_log_avg_dur_6'])
login_scene_offtime['c_scene_sczl_sum_avg_dur_6'] = login_scene_offtime['c_scene_sczl_sum_dur_6']/(login_scene_offtime['c_scene_tot_cnt_6']*login_scene_offtime['c_scene_log_avg_dur_6'])
login_scene_offtime['c_scene_sh_sum_avg_dur_6'] = login_scene_offtime['c_scene_sh_sum_dur_6']/(login_scene_offtime['c_scene_tot_cnt_6']*login_scene_offtime['c_scene_log_avg_dur_6'])

login_scene_offtime['c_scene_reg_sum_avg_dur_5'] = login_scene_offtime['c_scene_reg_sum_dur_5']/(login_scene_offtime['c_scene_tot_cnt_5']*login_scene_offtime['c_scene_log_avg_dur_5'])
login_scene_offtime['c_scene_dl_sum_avg_dur_5'] = login_scene_offtime['c_scene_dl_sum_dur_5']/(login_scene_offtime['c_scene_tot_cnt_5']*login_scene_offtime['c_scene_log_avg_dur_5'])
login_scene_offtime['c_scene_od_sum_avg_dur_5'] = login_scene_offtime['c_scene_od_sum_dur_5']/(login_scene_offtime['c_scene_tot_cnt_5']*login_scene_offtime['c_scene_log_avg_dur_5'])
login_scene_offtime['c_scene_rp_sum_avg_dur_5'] = login_scene_offtime['c_scene_rp_sum_dur_5']/(login_scene_offtime['c_scene_tot_cnt_5']*login_scene_offtime['c_scene_log_avg_dur_5'])
login_scene_offtime['c_scene_xgxx_sum_avg_dur_5'] = login_scene_offtime['c_scene_xgxx_sum_dur_5']/(login_scene_offtime['c_scene_tot_cnt_5']*login_scene_offtime['c_scene_log_avg_dur_5'])
login_scene_offtime['c_scene_plsp_sum_avg_dur_5'] = login_scene_offtime['c_scene_plsp_sum_dur_5']/(login_scene_offtime['c_scene_tot_cnt_5']*login_scene_offtime['c_scene_log_avg_dur_5'])
login_scene_offtime['c_scene_sczl_sum_avg_dur_5'] = login_scene_offtime['c_scene_sczl_sum_dur_5']/(login_scene_offtime['c_scene_tot_cnt_5']*login_scene_offtime['c_scene_log_avg_dur_5'])
login_scene_offtime['c_scene_sh_sum_avg_dur_5'] = login_scene_offtime['c_scene_sh_sum_dur_5']/(login_scene_offtime['c_scene_tot_cnt_5']*login_scene_offtime['c_scene_log_avg_dur_5'])

login_scene_offtime = login_scene_offtime.fillna(0)

login_scene_offtime_var = login_scene_offtime.var().reset_index()
login_scene_offtime_var.columns = ['index','val']

login_scene_offtime = login_scene_offtime.drop(login_scene_mdl_var[login_scene_mdl_var.val<0.1]['index'],axis=1)

login_scene_offtime.to_csv(r'features/login_scene_offtime.csv',index=False)

