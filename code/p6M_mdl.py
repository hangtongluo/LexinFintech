# -- coding: utf-8 --
import pandas as pd
import numpy as np

p6M_mdl = pd.read_csv(r"data\lexin_train\p6M_mdl.csv")
###############################################################################
#p6M_mdl['futilization01'] = p6M_mdl['futilization'].apply(lambda x: 0 if x == 0 else 1)
#p6M_mdl['cpt_pymt_rate'] = 1.0 * p6M_mdl['cpt_pymt'] / p6M_mdl['credit_limit']


###############################################################################
import datetime as dt
p6M_mdl['cyc_date'] = pd.to_datetime(p6M_mdl['cyc_date'],format='%Y-%m-%d %H:%M:%S')
p6M_mdl['cyc_date']=[i.month for i in p6M_mdl["cyc_date"]]

#去掉观测月这一列，起码现在觉得在这个表格没太大用
p6M_mdl=p6M_mdl.drop('pyear_month',axis=1)
p6M_mdl=p6M_mdl.drop('fcredit_update_time',axis=1)

#应该先弄好每个人有的六行数据的每个特征的最大值，最小值，均值，方差，中位数，众值
p6M_mdl_mean = p6M_mdl.groupby(['fuid_md5']).agg(np.mean)
p6M_mdl_max = p6M_mdl.groupby(['fuid_md5']).agg(np.max)
p6M_mdl_min = p6M_mdl.groupby(['fuid_md5']).agg(np.min)
p6M_mdl_std = p6M_mdl.groupby(['fuid_md5']).agg(np.std)
p6M_mdl_median = p6M_mdl.groupby(['fuid_md5']).agg(np.median)

p6M_mdl_mean = p6M_mdl_mean.reset_index()
p6M_mdl_max = p6M_mdl_max.reset_index()
p6M_mdl_min = p6M_mdl_min.reset_index()
p6M_mdl_std = p6M_mdl_std.reset_index()
p6M_mdl_median = p6M_mdl_median.reset_index()

p6M_mdl_mean = p6M_mdl_mean.drop('cyc_date',axis=1)
p6M_mdl_max = p6M_mdl_max.drop('cyc_date',axis=1)
p6M_mdl_min = p6M_mdl_min.drop('cyc_date',axis=1)
p6M_mdl_std = p6M_mdl_std.drop('cyc_date',axis=1)
p6M_mdl_median = p6M_mdl_median.drop('cyc_date',axis=1)

#将均值，最大值，最小值，方差，中值拼接起来，应该是53*5+1列数据
p6M_mdl_sta = pd.merge(p6M_mdl_mean,p6M_mdl_max,how='left',on ='fuid_md5',suffixes=['_mean', '_max'])
p6M_mdl_sta = pd.merge(p6M_mdl_sta,p6M_mdl_min,how='left',on ='fuid_md5',suffixes=['', '_min'])
p6M_mdl_sta = pd.merge(p6M_mdl_sta,p6M_mdl_std,how='left',on ='fuid_md5',suffixes=['', '_std'])
p6M_mdl_sta = pd.merge(p6M_mdl_sta,p6M_mdl_median,how='left',on ='fuid_md5',suffixes=['', '_median'])

#获取5,6,7,8,9,10月份数据
p6M_mdl_10 = p6M_mdl[p6M_mdl['cyc_date']==10]
p6M_mdl_10 = p6M_mdl_10.drop('cyc_date',axis=1)

p6M_mdl_9 = p6M_mdl[p6M_mdl['cyc_date']==9]
p6M_mdl_9 = p6M_mdl_9.drop('cyc_date',axis=1)

p6M_mdl_8 = p6M_mdl[p6M_mdl['cyc_date']==8]
p6M_mdl_8 = p6M_mdl_8.drop('cyc_date',axis=1)

p6M_mdl_7 = p6M_mdl[p6M_mdl['cyc_date']==7]
p6M_mdl_7 = p6M_mdl_7.drop('cyc_date',axis=1)

p6M_mdl_6 = p6M_mdl[p6M_mdl['cyc_date']==6]
p6M_mdl_6 = p6M_mdl_6.drop('cyc_date',axis=1)

p6M_mdl_5 = p6M_mdl[p6M_mdl['cyc_date']==5]
p6M_mdl_5 = p6M_mdl_5.drop('cyc_date',axis=1)

#将月份数据拼接起来
p6M_mdl_mon = pd.merge(p6M_mdl_10,p6M_mdl_9,how='left',on ='fuid_md5',suffixes=['_10', '_9'])
p6M_mdl_mon = pd.merge(p6M_mdl_mon,p6M_mdl_8,how='left',on ='fuid_md5',suffixes=['', '_8'])
p6M_mdl_mon = pd.merge(p6M_mdl_mon,p6M_mdl_7,how='left',on ='fuid_md5',suffixes=['', '_7'])
p6M_mdl_mon = pd.merge(p6M_mdl_mon,p6M_mdl_6,how='left',on ='fuid_md5',suffixes=['', '_6'])
p6M_mdl_mon = pd.merge(p6M_mdl_mon,p6M_mdl_5,how='left',on ='fuid_md5',suffixes=['', '_5'])

#再将上面两种数据拼接，一种是统计数据，一种是每月的数据
p6M_mdl = pd.merge(p6M_mdl_mon,p6M_mdl_sta,how='left',on ='fuid_md5')

login_scene_mdl_var = p6M_mdl.var().reset_index()
login_scene_mdl_var.columns = ['index','val']
p6M_mdl = p6M_mdl.drop(login_scene_mdl_var[login_scene_mdl_var.val<0.1]['index'],axis=1)
p6M_mdl.to_csv(r'features/p6M_mdl_sta.csv',index=False)


#测试集
p6M_offtime = pd.read_csv(r"data\lexin_test\p6M_offtime.csv")
###############################################################################
#p6M_offtime['futilization01'] = p6M_offtime['futilization'].apply(lambda x: 0 if x == 0 else 1)
#p6M_offtime['cpt_pymt_rate'] = 1.0 * p6M_offtime['cpt_pymt'] / p6M_offtime['credit_limit']


###############################################################################
import datetime as dt
p6M_offtime['cyc_date'] = pd.to_datetime(p6M_offtime['cyc_date'],format='%Y-%m-%d %H:%M:%S')
p6M_offtime['cyc_date']=[i.month for i in p6M_offtime["cyc_date"]]

p6M_offtime = p6M_offtime.drop('pyear_month',axis=1)
p6M_offtime=p6M_offtime.drop('fcredit_update_time',axis=1)

#应该先弄好每个人有的六行数据的每个特征的最大值，最小值，均值，方差，中位数，众值
p6M_offtime_mean = p6M_offtime.groupby(['fuid_md5']).agg(np.mean)
p6M_offtime_max = p6M_offtime.groupby(['fuid_md5']).agg(np.max)
p6M_offtime_min = p6M_offtime.groupby(['fuid_md5']).agg(np.min)
p6M_offtime_std = p6M_offtime.groupby(['fuid_md5']).agg(np.std)
p6M_offtime_median = p6M_offtime.groupby(['fuid_md5']).agg(np.median)

p6M_offtime_mean = p6M_offtime_mean.reset_index()
p6M_offtime_max = p6M_offtime_max.reset_index()
p6M_offtime_min = p6M_offtime_min.reset_index()
p6M_offtime_std = p6M_offtime_std.reset_index()
p6M_offtime_median = p6M_offtime_median.reset_index()

p6M_offtime_mean = p6M_offtime_mean.drop('cyc_date',axis=1)
p6M_offtime_max = p6M_offtime_max.drop('cyc_date',axis=1)
p6M_offtime_min = p6M_offtime_min.drop('cyc_date',axis=1)
p6M_offtime_std = p6M_offtime_std.drop('cyc_date',axis=1)
p6M_offtime_median = p6M_offtime_median.drop('cyc_date',axis=1)

#将均值，最大值，最小值，方差，中值拼接起来，应该是53*5+1列数据
p6M_offtime_sta = pd.merge(p6M_offtime_mean,p6M_offtime_max,how='left',on ='fuid_md5',suffixes=['_mean', '_max'])
p6M_offtime_sta = pd.merge(p6M_offtime_sta,p6M_offtime_min,how='left',on ='fuid_md5',suffixes=['', '_min'])
p6M_offtime_sta = pd.merge(p6M_offtime_sta,p6M_offtime_std,how='left',on ='fuid_md5',suffixes=['', '_std'])
p6M_offtime_sta = pd.merge(p6M_offtime_sta,p6M_offtime_median,how='left',on ='fuid_md5',suffixes=['', '_median'])

#获取7,8,9,10,11,12月份数据
p6M_offtime_12 = p6M_offtime[p6M_offtime['cyc_date']==12]
p6M_offtime_12 = p6M_offtime_12.drop('cyc_date',axis=1)

p6M_offtime_11 = p6M_offtime[p6M_offtime['cyc_date']==11]
p6M_offtime_11 = p6M_offtime_11.drop('cyc_date',axis=1)

p6M_offtime_10 = p6M_offtime[p6M_offtime['cyc_date']==10]
p6M_offtime_10 = p6M_offtime_10.drop('cyc_date',axis=1)

p6M_offtime_9 = p6M_offtime[p6M_offtime['cyc_date']==9]
p6M_offtime_9 = p6M_offtime_9.drop('cyc_date',axis=1)

p6M_offtime_8 = p6M_offtime[p6M_offtime['cyc_date']==8]
p6M_offtime_8 = p6M_offtime_8.drop('cyc_date',axis=1)

p6M_offtime_7 = p6M_offtime[p6M_offtime['cyc_date']==7]
p6M_offtime_7 = p6M_offtime_7.drop('cyc_date',axis=1)

p6M_offtime_mon = pd.merge(p6M_offtime_12,p6M_offtime_11,how='left',on ='fuid_md5',suffixes=['_10', '_9'])
p6M_offtime_mon = pd.merge(p6M_offtime_mon,p6M_offtime_10,how='left',on ='fuid_md5',suffixes=['', '_8'])
p6M_offtime_mon = pd.merge(p6M_offtime_mon,p6M_offtime_9,how='left',on ='fuid_md5',suffixes=['', '_7'])
p6M_offtime_mon = pd.merge(p6M_offtime_mon,p6M_offtime_8,how='left',on ='fuid_md5',suffixes=['', '_6'])
p6M_offtime_mon = pd.merge(p6M_offtime_mon,p6M_offtime_7,how='left',on ='fuid_md5',suffixes=['', '_5'])

p6M_offtime = pd.merge(p6M_offtime_mon,p6M_offtime_sta,how='left',on ='fuid_md5')

p6M_offtime = p6M_offtime.drop(login_scene_mdl_var[login_scene_mdl_var.val<0.1]['index'],axis=1)
p6M_offtime.to_csv(r'features/p6M_offtime_sta.csv',index=False)





















