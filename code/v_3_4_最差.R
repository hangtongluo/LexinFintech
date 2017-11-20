rm(list = ls())
gc()
setwd("D:\\RstudioWorkplace\\牛客比赛") #运行前改为当前路径！！！！！！！！！！！！！！！！
options(max.print = 1500000)
library(data.table)
library(magrittr)
library(lubridate)
library(stringr)
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
mode_max = function(x) {
  as.numeric(names(table(x))[table(x)==max(table(x))][1])
}
mode_min = function(x) {
  as.numeric(names(table(x))[table(x)==min(table(x))][1])
}

train_user<- fread(".\\data\\lexin_train\\ud_mdl.csv",encoding="UTF-8")

test_user<- fread(".\\data\\lexin_test\\ud_offtime.csv",encoding="UTF-8")

train_2<- fread(".\\data\\lexin_train\\login_scene_mdl.csv",encoding="UTF-8")
dim(train_2)

test_2<- fread(".\\data\\lexin_test\\login_scene_offtime.csv",encoding="UTF-8")
dim(test_2)

##train与test用户有交集，1011个，这几个不用预测
train_user[which(train_user$fuid_md5 %in% test_user$fuid_md5)][fuid_md5=="391b2517397154209f84646bff735711"]
test_user[which(test_user$fuid_md5 %in% train_user$fuid_md5)][fuid_md5=="391b2517397154209f84646bff735711"]

train_3<- fread(".\\data\\lexin_train\\od_in6m_mdl.csv",encoding="UTF-8",fill = TRUE)
train_3$faccount_time <- todate(train_3$faccount_time)
dim(train_3)
#max(train_3$faccount_time)
test_3 <- fread(".\\data\\lexin_test\\od_in6m_offtime.csv",encoding="UTF-8",fill = TRUE)

train_4<- fread(".\\data\\lexin_train\\p6M_mdl.csv",encoding="UTF-8")

dim(train_4)
#min(test_4$cyc_date[test_4$cyc_date> ymd("2015-01-01")])
test_4<- fread(".\\data\\lexin_test\\p6M_offtime.csv",encoding="UTF-8")


train_5<- fread(".\\data\\lexin_train\\p12M_mdl.csv",encoding="UTF-8")

test_5 <- fread(".\\data\\lexin_test\\p12M_offtime.csv",encoding="UTF-8")


train_6 <- fread(".\\data\\lexin_train\\dep_mdl.csv",encoding="UTF-8")
train_6[fuid_md5 %in% train_user[which(train_user$fuid_md5 %in% test_user$fuid_md5)]$fuid_md5,.(dep)]%>%sum(.$dep)

train5_pro <-train_5[,.(od_brw_12_mean = mean(od_brw,na.rm = T)),by = .(fuid_md5)]
train5_pro
train3_pro<- train_3[,.(buy_num = .N
                        ,forder_type_modemax = mode_max(forder_type)
                        ,forder_type_modemin = mode_min(forder_type)
                        ,fsub_order_type_modemax = mode_max(fsub_order_type)
                        ,fsub_order_type_modemin = mode_min(fsub_order_type)
                        ,fsale_type_modemax = mode_max(fsale_type)
                        ,fsale_type_modemin = mode_min(fsale_type)
                        ,ftotal_amount_max = max(ftotal_amount)
                        ,ftotal_amount_min = min(ftotal_amount)
                        ,ftotal_amount_avg = mean(ftotal_amount,na.rm = T)
                        #,ftotal_amount_median = median(ftotal_amount)
                        ,ftotal_firstpay_max = max(ftotal_firstpay)
                        ,ftotal_firstpay_min = min(ftotal_firstpay)
                        ,ftotal_firstpay_avg = mean(ftotal_firstpay,na.rm = T)
                        #,ftotal_firstpay_median = median(ftotal_firstpay)
                        ,ffirstpay_fee_type_mode_max = mode_max(ffirstpay_fee_type)
                        ,ffirstpay_fee_type_mode_min = mode_min(ffirstpay_fee_type)
                        ,fmax_fq_num_max = max(fmax_fq_num)
                        ,fmax_fq_num_min = min(fmax_fq_num)
                        ,fmax_fq_num_avg = mean(fmax_fq_num,na.rm = T)
                        #,fmax_fq_num_median = median(fmax_fq_num)
                        ,fmax_fq_num_modemax = mode_max(fmax_fq_num)
                        ,fmax_fq_num_modemin = mode_min(fmax_fq_num)
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
train_all
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
test5_pro
test3_pro<- test_3[,.(buy_num = .N
                      ,forder_type_modemax = mode_max(forder_type)
                      ,forder_type_modemin = mode_min(forder_type)
                      ,fsub_order_type_modemax = mode_max(fsub_order_type)
                      ,fsub_order_type_modemin = mode_min(fsub_order_type)
                      ,fsale_type_modemax = mode_max(fsale_type)
                      ,fsale_type_modemin = mode_min(fsale_type)
                      ,ftotal_amount_max = max(ftotal_amount)
                      ,ftotal_amount_min = min(ftotal_amount)
                      ,ftotal_amount_avg = mean(ftotal_amount,na.rm = T)
                      #,ftotal_amount_median = median(ftotal_amount)
                      ,ftotal_firstpay_max = max(ftotal_firstpay)
                      ,ftotal_firstpay_min = min(ftotal_firstpay)
                      ,ftotal_firstpay_avg = mean(ftotal_firstpay,na.rm = T)
                      #,ftotal_firstpay_median = median(ftotal_firstpay)
                      ,ffirstpay_fee_type_mode_max = mode_max(ffirstpay_fee_type)
                      ,ffirstpay_fee_type_mode_min = mode_min(ffirstpay_fee_type)
                      ,fmax_fq_num_max = max(fmax_fq_num)
                      ,fmax_fq_num_min = min(fmax_fq_num)
                      ,fmax_fq_num_avg = mean(fmax_fq_num,na.rm = T)
                      #,fmax_fq_num_median = median(fmax_fq_num)
                      ,fmax_fq_num_modemax = mode_max(fmax_fq_num)
                      ,fmax_fq_num_modemin = mode_min(fmax_fq_num)
),by = (fuid_md5)]



a2 <- merge(test_user,test5_pro,by = 'fuid_md5')%>%
  merge(test3_pro,by = 'fuid_md5',all.x = T)

#-------------------------------方案一：一个月一个月建模-----------------------------
##------------------------------先用10月的建模-------------------------------------
b2 <- merge(test_4[cyc_date>ymd("2016-12-01"),],
            test_2[cyc_date>ymd("2016-12-01"),],
            by = c("fuid_md5",'cyc_date'),all.x = T)
dim(b2)
test_all <- merge(a2,b2,by = "fuid_md5",all.x = T)
dim(test_all)
dim(train_all)

test_all[,":=" (dep = NULL
                ,actual_od_brw_f6m = NULL
                ,fregister_time = todate(fregister_time)
                ,fpocket_auth_time = todate(fpocket_auth_time)
                ,fcal_graduation = todate(fcal_graduation)
                ,pyear_month.x = todate(pyear_month.x)
                ,fcredit_update_time = todate(fcredit_update_time)
                ,pyear_month.y = todate(pyear_month.y)
)]

#----------------------------------模型训练-------------------------------------
library(xgboost)
set.seed(0)
params1 <- list(
  max.depth = 8
  ,eta = 0.01
  ,subsample=0.9
  ,colsample_bytree=0.9
  ,min_child_weight=30
  ,gamma=18
  #,objective = "reg:linear"
  ,eval_metric="auc"
  ,objective = 'binary:logistic'
  
) 
xgboost_fit_auc <- xgboost(params = params1,nrounds = 1000,data=data.matrix(train_all),
                           label = y1)

pre_xgboos_auc_01 <- predict(xgboost_fit_auc ,data.matrix(test_all))
# pre_xgboos_auc%>%min
# 
# pre_xgboos_auc_01 <- (pre_xgboos_auc-min(pre_xgboos_auc))/(max(pre_xgboos_auc)-min(pre_xgboos_auc))
# pre_xgboos_auc_01

names <- colnames(train_all)
importance_matrix <- xgb.importance(names, model = xgboost_fit_auc)
xgb.ggplot.importance(importance_matrix[1:50,])



set.seed(0)
params2 <- list(
  max.depth = 8
  ,eta = 0.01
  ,subsample=0.9
  ,colsample_bytree=0.9
  ,min_child_weight=30
  ,gamma=18
  #,objective = "reg:linear"
  ,eval_metric="mae"
  ,objective = 'reg:linear'
  
) 
xgboost_fit_mae <- xgboost(params = params2,nrounds = 1000,data=data.matrix(train_all),
                           label = y2)
importance_matrix2 <- xgb.importance(names, model = xgboost_fit_mae)
xgb.plot.importance(importance_matrix2[1:50,])

pre_xgboos_mae <- predict(xgboost_fit_mae ,data.matrix(test_all))
pre_xgboos_mae <- ifelse(pre_xgboos_mae<0,0,pre_xgboos_mae)

submission <- test_all[,.(fuid_md5,prob = pre_xgboos_auc_01,num = pre_xgboos_mae)]

submission2 <- merge(test_user[,.(fuid_md5)],submission,by = 'fuid_md5',all.x = T,sort = F)
submission2
fwrite(submission2,"./data/submission/combin/submission_v3_4.txt",col.names = F,sep = '\t')













xgboost(data = data.matrix(train[,-4]),
        label =as.numeric(train$ciiquantity_month),
        max.depth = 8, eta = 0.01,subsample=0.9,colsample_bytree=0.9,
        min_child_weight=30,gamma=18,
        nround =1200 ,objective = "reg:linear",eval_metric="rmse")
lightgbm_fit <- xgboost(params = params,nrounds = 100,data=data.matrix(train),
                        label = as.factor(label))




set.seed(0)
params2 <-  list(
  objective='binary',
  metric="binary_error",
  boosting = 'gbdt',
  learning_rate =0.01,
  num_leaves= 256,
  bagging_fraction = 0.7,
  bagging_freq = 0,
  bagging_seed = 0,
  feature_fraction=0.7,#随机抽取列
  feature_fraction_seed = 0,
  max_bin= 100,
  nthread = 10,
  max_depth= 8,
  # scale_pos_weight=37
  is_unbalance = T
)



















