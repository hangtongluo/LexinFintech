rm(list = ls())
gc()
setwd("D:\\RstudioWorkplace\\牛客比赛") #运行前改为当前路径
options(max.print = 1500000)
library(data.table)
library(magrittr)
library(lubridate)
library(stringr)
library(REmap)
todate <- function(date){
  day <- substring(date,1,2)
  month <- substring(date,3,5)
  year <- substring(date,6,7)
  month[month=='JAN'] = 01
  month[month=='FEB'] = 02
  month[month=='MAR'] = 03
  month[month=='APR'] = 04
  month[month=='MAY'] = 05
  month[month=='JUN'] = 06
  month[month=='JUL'] = 07
  month[month=='AUG'] = 08
  month[month=='SEP'] = 09
  month[month=='OCT'] = 10
  month[month=='NOV'] = 11
  month[month=='DEC'] = 12
  date <- lubridate::ymd(paste(year,month,day,sep = '-'))
  return(date)
}
# mode_max = function(x) {
#   as.numeric(names(table(x))[table(x)==max(table(x))][1])
# }
# mode_min = function(x) {
#   as.numeric(names(table(x))[table(x)==min(table(x))][1])
# }

train_user<- fread(".\\data\\lexin_train\\ud_mdl.csv",encoding="UTF-8")
# train_user
test_user<- fread(".\\data\\lexin_test\\ud_offtime.csv",encoding="UTF-8")
test_user1 <- test_user[,.(fuid_md5)]
# test_user
train_2<- fread(".\\data\\lexin_train\\login_scene_mdl.csv",encoding="UTF-8")
dim(train_2)
# train_2
test_2<- fread(".\\data\\lexin_test\\login_scene_offtime.csv",encoding="UTF-8")
dim(test_2)
# test_2
##train与test用户有交集，1011个，这几个不用预测
train_user[which(train_user$fuid_md5 %in% test_user$fuid_md5)][fuid_md5=="391b2517397154209f84646bff735711"]
test_user[which(test_user$fuid_md5 %in% train_user$fuid_md5)][fuid_md5=="391b2517397154209f84646bff735711"]

train_3<- fread(".\\data\\lexin_train\\od_in6m_mdl.csv",encoding="UTF-8",fill = TRUE)
train_3$faccount_time <- todate(train_3$faccount_time)
dim(train_3)
#max(train_3$faccount_time)
test_3 <- fread(".\\data\\lexin_test\\od_in6m_offtime.csv",encoding="UTF-8",fill = TRUE)
test_3$faccount_time <- todate(test_3$faccount_time)

train_4<- fread(".\\data\\lexin_train\\p6M_mdl.csv",encoding="UTF-8")
# train_4
dim(train_4)
#min(test_4$cyc_date[test_4$cyc_date> ymd("2015-01-01")])
test_4<- fread(".\\data\\lexin_test\\p6M_offtime.csv",encoding="UTF-8")
# test_4

train_5<- fread(".\\data\\lexin_train\\p12M_mdl.csv",encoding="UTF-8")
# train_5
test_5 <- fread(".\\data\\lexin_test\\p12M_offtime.csv",encoding="UTF-8")
# test_5

train_6 <- fread(".\\data\\lexin_train\\dep_mdl.csv",encoding="UTF-8")
train_6[fuid_md5 %in% train_user[which(train_user$fuid_md5 %in% test_user$fuid_md5)]$fuid_md5,.(dep)]%>%sum(.$dep)

#------------------------------------------加入经纬度信息-----------------------
provice<- c("浙江省","广东省","上海市","北京市","河北省","江苏省",
            "天津市","福建省","安徽省","河南省","辽宁省","吉林省",
            "山东省","黑龙江省","江西省","山西省","重庆市","陕西省",
            "内蒙古自治区","宁夏回族自治区","海南省","云南省",
            "广西壮族自治区","青海省","湖南省","湖北省","甘肃省",
            "四川省","贵州省","新疆维吾尔自治区","西藏自治区")
provice_lon_lat <- REmap::get_geo_position(provice)%>%as.data.table()
provice_lon_lat$city_num <- as.numeric(as.factor(provice_lon_lat$city))

train_user$city_num <- as.numeric(as.factor(train_user$fdomicile_provice))-1
train_user <- merge(train_user,provice_lon_lat[,.(lon,lat,city_num)],by = "city_num",all.x = T)
test_user$city_num <- as.numeric(as.factor(test_user$fdomicile_provice))-1
test_user <- merge(test_user,provice_lon_lat[,.(lon,lat,city_num)],by = "city_num",all.x = T)



train5_pro <-train_5[,.(od_brw_12_mean = mean(od_brw,na.rm = T)),by = .(fuid_md5)]
# train5_pro
train3_pro<- train_3[,.(buy_num = .N
                        # ,forder_type_modemax = mode_max(forder_type)
                        # ,forder_type_modemin = mode_min(forder_type)
                        # ,fsub_order_type_modemax = mode_max(fsub_order_type)
                        # ,fsub_order_type_modemin = mode_min(fsub_order_type)
                        # ,fsale_type_modemax = mode_max(fsale_type)
                        # ,fsale_type_modemin = mode_min(fsale_type)
                        ,ftotal_amount_max = max(ftotal_amount,na.rm = T)
                        ,ftotal_amount_min = min(ftotal_amount,na.rm = T)
                        ,ftotal_amount_avg = mean(ftotal_amount,na.rm = T)
                        #,ftotal_amount_median = median(ftotal_amount)
                        ,ftotal_firstpay_max = max(ftotal_firstpay,na.rm = T)
                        ,ftotal_firstpay_min = min(ftotal_firstpay,na.rm = T)
                        ,ftotal_firstpay_avg = mean(ftotal_firstpay,na.rm = T)
                        #,ftotal_firstpay_median = median(ftotal_firstpay)
                        # ,ffirstpay_fee_type_mode_max = mode_max(ffirstpay_fee_type)
                        # ,ffirstpay_fee_type_mode_min = mode_min(ffirstpay_fee_type)
                        ,fmax_fq_num_max = max(fmax_fq_num,na.rm = T)
                        ,fmax_fq_num_min = min(fmax_fq_num,na.rm = T)
                        ,fmax_fq_num_avg = mean(fmax_fq_num,na.rm = T)
                        #,fmax_fq_num_median = median(fmax_fq_num)
                        # ,fmax_fq_num_modemax = mode_max(fmax_fq_num)
                        # ,fmax_fq_num_modemin = mode_min(fmax_fq_num)
),by = (fuid_md5)]

train6_pro <- train_6[,.(fuid_md5,dep,actual_od_brw_f6m=actual_od_brw_f6m)]
a1 <- merge(train_user,train6_pro,by = 'fuid_md5')%>%merge(train5_pro,by = 'fuid_md5')%>%
  merge(train3_pro,by = 'fuid_md5',all.x = T)

#方案一：一个月一个月建模-----------------------------
##---------------------先用10月的建模
b1 <- merge(train_4[cyc_date>ymd("2016-10-01"),],
            train_2[cyc_date>ymd("2016-10-01"),],
            by = c("fuid_md5",'cyc_date'),all.x = T)

dim(b1)
train_all <- merge(a1,b1,by = "fuid_md5",all.x = T)
# train_all
dim(train_all)

y1 <- train_all$dep
y2 <- train_all$actual_od_brw_f6m
train_all[,":=" (dep = NULL
                 ,actual_od_brw_f6m = NULL
                 ,fregister_time = todate(fregister_time)
                 ,fpocket_auth_time = todate(fpocket_auth_time)
                 ,fcal_graduation = todate(fcal_graduation)
                 ,pyear_month.x = todate(pyear_month.x)
                 ,fcredit_update_time = todate(fcredit_update_time)
                 ,pyear_month.y = todate(pyear_month.y)
)]
dim(train_all)
#--------------------------------------------test-------------------------------------
test5_pro <-test_5[,.(od_brw_12_mean = mean(od_brw,na.rm = T)),by = .(fuid_md5)]
# test5_pro
test3_pro<- test_3[,.(buy_num = .N
                      # ,forder_type_modemax = mode_max(forder_type)
                      # ,forder_type_modemin = mode_min(forder_type)
                      # ,fsub_order_type_modemax = mode_max(fsub_order_type)
                      # ,fsub_order_type_modemin = mode_min(fsub_order_type)
                      # ,fsale_type_modemax = mode_max(fsale_type)
                      # ,fsale_type_modemin = mode_min(fsale_type)
                      ,ftotal_amount_max = max(ftotal_amount,na.rm = T)
                      ,ftotal_amount_min = min(ftotal_amount,na.rm = T)
                      ,ftotal_amount_avg = mean(ftotal_amount,na.rm = T)
                      #,ftotal_amount_median = median(ftotal_amount)
                      ,ftotal_firstpay_max = max(ftotal_firstpay,na.rm = T)
                      ,ftotal_firstpay_min = min(ftotal_firstpay,na.rm = T)
                      ,ftotal_firstpay_avg = mean(ftotal_firstpay,na.rm = T)
                      #,ftotal_firstpay_median = median(ftotal_firstpay)
                      # ,ffirstpay_fee_type_mode_max = mode_max(ffirstpay_fee_type)
                      # ,ffirstpay_fee_type_mode_min = mode_min(ffirstpay_fee_type)
                      ,fmax_fq_num_max = max(fmax_fq_num,na.rm = T)
                      ,fmax_fq_num_min = min(fmax_fq_num,na.rm = T)
                      ,fmax_fq_num_avg = mean(fmax_fq_num,na.rm = T)
                      #,fmax_fq_num_median = median(fmax_fq_num)
                      # ,fmax_fq_num_modemax = mode_max(fmax_fq_num)
                      # ,fmax_fq_num_modemin = mode_min(fmax_fq_num)
),by = (fuid_md5)]



a2 <- merge(test_user,test5_pro,by = 'fuid_md5')%>%
  merge(test3_pro,by = 'fuid_md5',all.x = T)

#-------------------------------方案一：一个月一个月建模-----------------------------
##------------------------------先用10月的建模-------------------------------------
b2 <- merge(test_4[cyc_date>ymd("2016-12-01"),],
            test_2[cyc_date>ymd("2016-12-01"),],
            by = c("fuid_md5",'cyc_date'),all.x = T)
# b2
dim(b2)
test_all <- merge(a2,b2,by = "fuid_md5",all.x = T)
# test_all
test_all[,":=" (
  fregister_time = todate(fregister_time)
  ,fpocket_auth_time = todate(fpocket_auth_time)
  ,fcal_graduation = todate(fcal_graduation)
  ,pyear_month.x = todate(pyear_month.x)
  ,fcredit_update_time = todate(fcredit_update_time)
  ,pyear_month.y = todate(pyear_month.y)
)]

dim(test_all)
dim(train_all)

#--------------------------------------初步清洗------------------------------------
# train_all[,]


#--------------------------------------初步造特征------------------------------------
train_all[,":=" (
  futilization_1_01 = ifelse(futilization>1,1,0)
  ,futilization_0_01 = ifelse(futilization>0,1,0)
  ,xizang_01 = ifelse(fdomicile_provice=='西藏自治区',1,0)
  ,xinjiang_01 = ifelse(fdomicile_provice=='新疆维吾尔自治区',1,0)
  ,guizhou_01 = ifelse(fdomicile_provice=='贵州省',1,0)
  ,sichuan_01 = ifelse(fdomicile_provice=='四川省',1,0)
  ,gansu_01 = ifelse(fdomicile_provice=='甘肃省',1,0)
  ,hubei_01 = ifelse(fdomicile_provice=='湖北省',1,0)
  ,hunan_01 = ifelse(fdomicile_provice=='湖南省',1,0)
  ,zhejiang_01 = ifelse(fdomicile_provice=='浙江省',1,0)
  ,guangdong_01 = ifelse(fdomicile_provice=='广东省',1,0)
  ,shanghai_01 = ifelse(fdomicile_provice=='上海市',1,0)
  ,beijing_01 = ifelse(fdomicile_provice=='北京市',1,0)
  ,fpocket_auth_time_sub_fregister_time = as.numeric(fpocket_auth_time - fregister_time) #授信时间减去注册时间
  ,fcredit_update_time_sub_fregister_time = as.numeric(fcredit_update_time - fregister_time)
  ,fcredit_update_time_sub_fpocket_auth_time = as.numeric(fcredit_update_time - fpocket_auth_time)
  ,zdfq_paying_mon_fee_dev_zdfq_paying_tot_fee = zdfq_paying_mon_fee/(zdfq_paying_tot_fee+0.0000001)
  ,zdfq_paying_tot_fee_dev_zdfq_paying_mon_fee = zdfq_paying_tot_fee/(zdfq_paying_mon_fee+0.0000001)
  ,zdfq_bal_dev_ds3c_bal = zdfq_bal / (ds3c_bal+0.0000001)
  ,zdfq_bal_dev_bal = zdfq_bal / (bal+0.0000001)
  ,fdomicile_provice_01 = ifelse(fdomicile_provice == sch_fprovince_name,1,0) #籍贯省份与学校省份
  ,fuid_md5 = NULL
  ,fschoolarea_name_md5 =NULL
  ,fdomicile_provice = NULL
  ,fdomicile_city = NULL
  ,fdomicile_area = NULL
  ,sch_fprovince_name = NULL
  ,sch_fcity_name = NULL
  ,sch_fregion_name = NULL
  ,sch_fcompany_name = NULL
  #3c类特征
  ,od_3c_cnt_01 = ifelse(od_3c_cnt>0,1,0)
  ,od_3c_brw_01 = ifelse(od_3c_brw>0,1,0)
  ,cumu_od_3c_cnt_01 = ifelse(cumu_od_3c_cnt>0,1,0)
  ,cumu_od_3c_brw_01 = ifelse(cumu_od_3c_brw>0,1,0)
  ,payed_3c_capital_01 = ifelse(payed_3c_capital>0,1,0)
  ,payed_3c_mon_fee_01 = ifelse(payed_3c_mon_fee>0,1,0)
  ,payed_3c_tot_fee_01 = ifelse(payed_3c_tot_fee>0,1,0)
  ,ds3c_bal_01 = ifelse(ds3c_bal>0,1,0)
  ,ds3c_paying_mon_fee_01 = ifelse(ds3c_paying_mon_fee>0,1,0)
  ,ds3c_paying_tot_fee_01 = ifelse(ds3c_paying_tot_fee>0,1,0)
  
  ,od_brw_devide_od_cnt = od_brw / (od_cnt +0.0000001)
  ,actual_od_brw_devide_actual_od_cnt = actual_od_brw / (actual_od_cnt   +0.0000001)
  ,virtual_od_brw_devide_virtual_od_cnt = virtual_od_brw / (virtual_od_cnt  +0.0000001)
  ,od_3c_brw_devide_od_3c_cnt = od_3c_brw / (od_3c_cnt       +0.0000001)
  ,od_bh_brw_devide_od_bh_cnt = od_bh_brw / (od_bh_cnt       +0.0000001)
  ,od_yl_brw_devide_od_yl_cnt = od_yl_brw / (od_yl_cnt       +0.0000001)
  ,od_xj_brw_devide_od_xj_cnt = od_xj_brw / (od_xj_cnt       +0.0000001)
  ,od_ptsh_brw_devide_od_ptsh_cnt = od_ptsh_brw / (od_ptsh_cnt     +0.0000001)
  ,od_zdfq_brw_devide_od_zdfq_cnt = od_zdfq_brw / (od_zdfq_cnt     +0.0000001)
  ,od_xssh_brw_devide_od_xssh_cnt = od_xssh_brw / (od_xssh_cnt     +0.0000001)
  ,od_zdyq_brw_devide_od_zdyq_cnt = od_zdyq_brw / (od_zdyq_cnt     +0.0000001)
  ,od_lh_new_brw_devide_od_lh_new_cnt = od_lh_new_brw / (od_lh_new_cnt   +0.0000001)
  
  ,cumu_od_brw_devide_cumu_od_cnt = cumu_od_brw / (cumu_od_cnt+0.0000001)
  ,cumu_actual_od_brw_devide_cumu_actual_od_cnt = cumu_actual_od_brw / (cumu_actual_od_cnt+0.0000001)
  ,cumu_virtual_od_brw_devide_cumu_virtual_od_cnt = cumu_virtual_od_brw / (cumu_virtual_od_cnt+0.0000001)
  ,cumu_od_3c_brw_devide_cumu_od_3c_cnt = cumu_od_3c_brw / (cumu_od_3c_cnt+0.0000001)
  ,cumu_od_bh_brw_devide_cumu_od_bh_cnt = cumu_od_bh_brw / (cumu_od_bh_cnt+0.0000001)
  ,cumu_od_yl_brw_devide_cumu_od_yl_cnt = cumu_od_yl_brw / (cumu_od_yl_cnt+0.0000001)
  ,cumu_od_xj_brw_devide_cumu_od_xj_cnt = cumu_od_xj_brw / (cumu_od_xj_cnt+0.0000001)
  ,cumu_od_ptsh_brw_devide_cumu_od_ptsh_cnt = cumu_od_ptsh_brw / (cumu_od_ptsh_cnt+0.0000001)
  ,cumu_od_zdfq_brw_devide_cumu_od_zdfq_cnt = cumu_od_zdfq_brw / (cumu_od_zdfq_cnt+0.0000001)
  ,cumu_od_xssh_brw_devide_cumu_od_xssh_cnt = cumu_od_xssh_brw / (cumu_od_xssh_cnt+0.0000001)
  ,cumu_od_zdyq_brw_devide_cumu_od_zdyq_cnt = cumu_od_zdyq_brw / (cumu_od_zdyq_cnt+0.0000001)
  ,cumu_od_lh_new_brw_devide_cumu_od_lh_new_cnt = cumu_od_lh_new_brw / (cumu_od_lh_new_cnt+0.0000001)
  
  ,payed_capital_devide_payed_tot_fee = payed_capital / (payed_tot_fee+0.00000001)
  ,payed_3c_capital_devide_payed_3c_tot_fee = payed_3c_capital / (payed_3c_tot_fee+0.00000001)
  ,payed_bh_capital_devide_payed_bh_tot_fee = payed_bh_capital / (payed_bh_tot_fee+0.00000001)
  ,payed_yl_capital_devide_payed_yl_tot_fee = payed_yl_capital / (payed_yl_tot_fee+0.00000001)
  ,payed_xj_capital_devide_payed_xj_tot_fee = payed_xj_capital / (payed_xj_tot_fee+0.00000001)
  ,payed_ptsh_capital_devide_payed_ptsh_tot_fee = payed_ptsh_capital / (payed_ptsh_tot_fee+0.00000001)
  ,payed_zdfq_capital_devide_payed_zdfq_tot_fee = payed_zdfq_capital / (payed_zdfq_tot_fee+0.00000001)
  ,payed_xssh_capital_devide_payed_xssh_tot_fee = payed_xssh_capital / (payed_xssh_tot_fee+0.00000001)
  ,payed_zdyq_capital_devide_payed_zdyq_tot_fee = payed_zdyq_capital / (payed_zdyq_tot_fee+0.00000001)
  ,payed_lh_new_capital_devide_payed_lh_new_tot_fee = payed_lh_new_capital / (payed_lh_new_tot_fee+0.00000001)
  
  ,paying_complete_od_cnt_sub_payed_complete_od_cnt = paying_complete_od_cnt - payed_complete_od_cnt
  ,paying_complete_od_brw_sub_payed_complete_od_brw = paying_complete_od_brw - payed_complete_od_brw
  # ,used = credit_limit * futilization
  # ,same = credit_limit-credit_limit*futilization -fopen_to_buy
  ,ratio = cpt_pymt/credit_limit
  # ,by = futilization*fopen_to_buy
  
)]

fuid_md5 <- test_all$fuid_md5
test_all[,":=" ( futilization_1_01 = ifelse(futilization>1,1,0)
                 ,futilization_0_01 = ifelse(futilization>0,1,0)
                 ,xizang_01 = ifelse(fdomicile_provice=='西藏自治区',1,0)
                 ,xinjiang_01 = ifelse(fdomicile_provice=='新疆维吾尔自治区',1,0)
                 ,guizhou_01 = ifelse(fdomicile_provice=='贵州省',1,0)
                 ,sichuan_01 = ifelse(fdomicile_provice=='四川省',1,0)
                 ,gansu_01 = ifelse(fdomicile_provice=='甘肃省',1,0)
                 ,hubei_01 = ifelse(fdomicile_provice=='湖北省',1,0)
                 ,hunan_01 = ifelse(fdomicile_provice=='湖南省',1,0)
                 ,zhejiang_01 = ifelse(fdomicile_provice=='浙江省',1,0)
                 ,guangdong_01 = ifelse(fdomicile_provice=='广东省',1,0)
                 ,shanghai_01 = ifelse(fdomicile_provice=='上海市',1,0)
                 ,beijing_01 = ifelse(fdomicile_provice=='北京市',1,0)
                 ,fpocket_auth_time_sub_fregister_time = as.numeric(fpocket_auth_time - fregister_time) #授信时间减去注册时间
                 ,fcredit_update_time_sub_fregister_time = as.numeric(fcredit_update_time - fregister_time)
                 ,fcredit_update_time_sub_fpocket_auth_time = as.numeric(fcredit_update_time - fpocket_auth_time)
                 ,zdfq_paying_mon_fee_dev_zdfq_paying_tot_fee = zdfq_paying_mon_fee/(zdfq_paying_tot_fee+0.0000001)
                 ,zdfq_paying_tot_fee_dev_zdfq_paying_mon_fee = zdfq_paying_tot_fee/(zdfq_paying_mon_fee+0.0000001)
                 ,zdfq_bal_dev_ds3c_bal = zdfq_bal / (ds3c_bal+0.0000001)
                 ,zdfq_bal_dev_bal = zdfq_bal / (bal+0.0000001)
                 ,fdomicile_provice_01 = ifelse(fdomicile_provice == sch_fprovince_name,1,0) #籍贯省份与学校省份
                 ,fuid_md5 = NULL
                 ,fschoolarea_name_md5 =NULL
                 ,fdomicile_provice = NULL
                 ,fdomicile_city = NULL
                 ,fdomicile_area = NULL
                 ,sch_fprovince_name = NULL
                 ,sch_fcity_name = NULL
                 ,sch_fregion_name = NULL
                 ,sch_fcompany_name = NULL
                 #3c类特征
                 ,od_3c_cnt_01 = ifelse(od_3c_cnt>0,1,0)
                 ,od_3c_brw_01 = ifelse(od_3c_brw>0,1,0)
                 ,cumu_od_3c_cnt_01 = ifelse(cumu_od_3c_cnt>0,1,0)
                 ,cumu_od_3c_brw_01 = ifelse(cumu_od_3c_brw>0,1,0)
                 ,payed_3c_capital_01 = ifelse(payed_3c_capital>0,1,0)
                 ,payed_3c_mon_fee_01 = ifelse(payed_3c_mon_fee>0,1,0)
                 ,payed_3c_tot_fee_01 = ifelse(payed_3c_tot_fee>0,1,0)
                 ,ds3c_bal_01 = ifelse(ds3c_bal>0,1,0)
                 ,ds3c_paying_mon_fee_01 = ifelse(ds3c_paying_mon_fee>0,1,0)
                 ,ds3c_paying_tot_fee_01 = ifelse(ds3c_paying_tot_fee>0,1,0)
                 ,od_brw_devide_od_cnt = od_brw / (od_cnt +0.0000001)
                 
                 ,actual_od_brw_devide_actual_od_cnt = actual_od_brw / (actual_od_cnt   +0.0000001)
                 ,virtual_od_brw_devide_virtual_od_cnt = virtual_od_brw / (virtual_od_cnt  +0.0000001)
                 ,od_3c_brw_devide_od_3c_cnt = od_3c_brw / (od_3c_cnt       +0.0000001)
                 ,od_bh_brw_devide_od_bh_cnt = od_bh_brw / (od_bh_cnt       +0.0000001)
                 ,od_yl_brw_devide_od_yl_cnt = od_yl_brw / (od_yl_cnt       +0.0000001)
                 ,od_xj_brw_devide_od_xj_cnt = od_xj_brw / (od_xj_cnt       +0.0000001)
                 ,od_ptsh_brw_devide_od_ptsh_cnt = od_ptsh_brw / (od_ptsh_cnt     +0.0000001)
                 ,od_zdfq_brw_devide_od_zdfq_cnt = od_zdfq_brw / (od_zdfq_cnt     +0.0000001)
                 ,od_xssh_brw_devide_od_xssh_cnt = od_xssh_brw / (od_xssh_cnt     +0.0000001)
                 ,od_zdyq_brw_devide_od_zdyq_cnt = od_zdyq_brw / (od_zdyq_cnt     +0.0000001)
                 ,od_lh_new_brw_devide_od_lh_new_cnt = od_lh_new_brw / (od_lh_new_cnt   +0.0000001)
                 
                 ,cumu_od_brw_devide_cumu_od_cnt = cumu_od_brw / (cumu_od_cnt+0.0000001)
                 ,cumu_actual_od_brw_devide_cumu_actual_od_cnt = cumu_actual_od_brw / (cumu_actual_od_cnt+0.0000001)
                 ,cumu_virtual_od_brw_devide_cumu_virtual_od_cnt = cumu_virtual_od_brw / (cumu_virtual_od_cnt+0.0000001)
                 ,cumu_od_3c_brw_devide_cumu_od_3c_cnt = cumu_od_3c_brw / (cumu_od_3c_cnt+0.0000001)
                 ,cumu_od_bh_brw_devide_cumu_od_bh_cnt = cumu_od_bh_brw / (cumu_od_bh_cnt+0.0000001)
                 ,cumu_od_yl_brw_devide_cumu_od_yl_cnt = cumu_od_yl_brw / (cumu_od_yl_cnt+0.0000001)
                 ,cumu_od_xj_brw_devide_cumu_od_xj_cnt = cumu_od_xj_brw / (cumu_od_xj_cnt+0.0000001)
                 ,cumu_od_ptsh_brw_devide_cumu_od_ptsh_cnt = cumu_od_ptsh_brw / (cumu_od_ptsh_cnt+0.0000001)
                 ,cumu_od_zdfq_brw_devide_cumu_od_zdfq_cnt = cumu_od_zdfq_brw / (cumu_od_zdfq_cnt+0.0000001)
                 ,cumu_od_xssh_brw_devide_cumu_od_xssh_cnt = cumu_od_xssh_brw / (cumu_od_xssh_cnt+0.0000001)
                 ,cumu_od_zdyq_brw_devide_cumu_od_zdyq_cnt = cumu_od_zdyq_brw / (cumu_od_zdyq_cnt+0.0000001)
                 ,cumu_od_lh_new_brw_devide_cumu_od_lh_new_cnt = cumu_od_lh_new_brw / (cumu_od_lh_new_cnt+0.0000001)
                 
                 ,payed_capital_devide_payed_tot_fee = payed_capital / (payed_tot_fee+0.00000001)
                 ,payed_3c_capital_devide_payed_3c_tot_fee = payed_3c_capital / (payed_3c_tot_fee+0.00000001)
                 ,payed_bh_capital_devide_payed_bh_tot_fee = payed_bh_capital / (payed_bh_tot_fee+0.00000001)
                 ,payed_yl_capital_devide_payed_yl_tot_fee = payed_yl_capital / (payed_yl_tot_fee+0.00000001)
                 ,payed_xj_capital_devide_payed_xj_tot_fee = payed_xj_capital / (payed_xj_tot_fee+0.00000001)
                 ,payed_ptsh_capital_devide_payed_ptsh_tot_fee = payed_ptsh_capital / (payed_ptsh_tot_fee+0.00000001)
                 ,payed_zdfq_capital_devide_payed_zdfq_tot_fee = payed_zdfq_capital / (payed_zdfq_tot_fee+0.00000001)
                 ,payed_xssh_capital_devide_payed_xssh_tot_fee = payed_xssh_capital / (payed_xssh_tot_fee+0.00000001)
                 ,payed_zdyq_capital_devide_payed_zdyq_tot_fee = payed_zdyq_capital / (payed_zdyq_tot_fee+0.00000001)
                 ,payed_lh_new_capital_devide_payed_lh_new_tot_fee = payed_lh_new_capital / (payed_lh_new_tot_fee+0.00000001)
                 
                 ,paying_complete_od_cnt_sub_payed_complete_od_cnt = paying_complete_od_cnt - payed_complete_od_cnt
                 ,paying_complete_od_brw_sub_payed_complete_od_brw = paying_complete_od_brw - payed_complete_od_brw
                 # ,used = credit_limit * futilization
                 # ,same = credit_limit-credit_limit*futilization -fopen_to_buy
                 ,ratio = cpt_pymt/credit_limit
                 # ,by = futilization*fopen_to_buy
)]



dim(train_all)
dim(test_all) 

library(lightgbm)
set.seed(0)
params1 <-  list(
  objective='binary',
  metric="auc",
  boosting = 'gbdt',
  learning_rate =0.01,
  num_leaves= 200,
  bagging_fraction = 0.7,
  bagging_freq = 0,
  bagging_seed = 0,
  feature_fraction=0.7,#随机抽取列
  feature_fraction_seed = 0,
  nthread = 4,
  max_depth= 6
  # scale_pos_weight=37
  #is_unbalance = T
)


lightgbm_fit <- lightgbm(params = params1,nrounds = 1000,data=data.matrix(train_all),
                         label = y1)


lightgbm_imp <- lgb.importance(lightgbm_fit, percentage = TRUE)
lgb.plot.importance(lightgbm_imp)
pre_lightgbm_auc <- predict(lightgbm_fit,data.matrix(test_all))

# pre_xgboos_mae <- train3_pro$ftotal_amount_avg*6
# 
# test3_pro[,.(fuid_md5,ftotal_amount_avg)]

submission <- test_all[,.(fuid_md5 = fuid_md5,prob = pre_lightgbm_auc)]

od_brw_6_mean <- test_5[ymd(cyc_date)>=ymd("2016-06-01"),.(od_brw_6_mean = mean(od_brw)),by = .(fuid_md5)]

submission <- merge(submission,od_brw_6_mean[,.(fuid_md5,mae = od_brw_6_mean *4)],by = "fuid_md5",all.x = T)
# submission$mae <- ifelse(is.na(submission$mae),0,submission$mae)

submission2 <- merge(test_user1[,.(fuid_md5)],submission,by = 'fuid_md5',all.x = T,sort = F)

submission2

fwrite(submission2,"./data/submission/combin/submission_v5_10.txt",col.names = F,sep = '\t')
