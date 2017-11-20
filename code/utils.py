# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 09:01:13 2017

@author: Administrator
"""

import gc
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb


#############################basis process functions###################################
#变量编码函数
def LabelEncoderTool(trainDF, testDF, usecol):
    alldata = pd.concat([trainDF, testDF])
    for col in usecol:
        le = LabelEncoder()
        le.fit(alldata[col].astype('str'))
        trainDF[col] = le.transform(trainDF[col].astype('str'))
        testDF[col] = le.transform(testDF[col].astype('str'))
    return trainDF, testDF
    

#日期数据转码函数
def todatetime(x):
    if x != np.nan:
        return datetime.strptime(x, '%d%b%y:%H:%M:%S')  #特定的数据格式
    else:
        return x

def ToDatetimeTool(DF, usecol):
    for col in usecol:
        DF[col] = DF[col].fillna('01JUL17:00:00:00').apply(todatetime)  #fcal_graduation字段存在个别nan值
    return DF


#############################features extract functions###################################
#分组统计函数
def groupby_unique_Tool(DF, key, colname):
    DF = DF.groupby(key)\
                .apply(lambda df: len(df[colname].unique()))\
                .reset_index()\
                .rename(columns={0:'%s_unique' % colname}) 
    return DF

def groupby_sum_Tool(DF, key, colname):
    DF = DF.groupby(key)\
                .apply(lambda df: df[colname].sum())\
                .reset_index()\
                .rename(columns={0:'%s_sum' % colname}) 
    return DF

def groupby_mean_Tool(DF, key, colname):
    DF = DF.groupby(key)\
                .apply(lambda df: df[colname].mean())\
                .reset_index()\
                .rename(columns={0:'%s_mean' % colname}) 
    return DF

def groupby_std_Tool(DF, key, colname):
    DF = DF.groupby(key)\
                .apply(lambda df: df[colname].std())\
                .reset_index()\
                .rename(columns={0:'%s_std' % colname}) 
    return DF

def groupby_max_Tool(DF, key, colname):
    DF = DF.groupby(key)\
                .apply(lambda df: df[colname].max())\
                .reset_index()\
                .rename(columns={0:'%s_max' % colname}) 
    return DF

def groupby_min_Tool(DF, key, colname):
    DF = DF.groupby(key)\
                .apply(lambda df: df[colname].min())\
                .reset_index()\
                .rename(columns={0:'%s_min' % colname}) 
    return DF

def groupby_sum_pivot_Tool(trainDF, key, colname):
    temp = trainDF.groupby(key)\
            .apply(lambda df: df[colname].sum())\
            .reset_index()\
            .rename(columns={0:'%s_sum' % key[1]})\
            .pivot(columns=key[1], values='%s_sum' % key[1], index=key[0])\
            .reset_index()
    cols = [temp.columns[0]] + ['%s_sum' % key[1]+'_'+str(x) for x in temp.columns[1:]]
    temp.columns = cols
    return temp
    
def groupby_mean_pivot_Tool(trainDF, key, colname):
    temp = trainDF.groupby(key)\
            .apply(lambda df: df[colname].mean())\
            .reset_index()\
            .rename(columns={0:'%s_mean' % key[1]})\
            .pivot(columns=key[1], values='%s_mean' % key[1], index=key[0])\
            .reset_index()
    cols = [temp.columns[0]] + ['%s_mean' % key[1]+'_'+str(x) for x in temp.columns[1:]]
    temp.columns = cols
    return temp 

def groupby_max_pivot_Tool(trainDF, key, colname):
    temp = trainDF.groupby(key)\
            .apply(lambda df: df[colname].max())\
            .reset_index()\
            .rename(columns={0:'%s_max' % key[1]})\
            .pivot(columns=key[1], values='%s_max' % key[1], index=key[0])\
            .reset_index()
    cols = [temp.columns[0]] + ['%s_max' % key[1]+'_'+str(x) for x in temp.columns[1:]]
    temp.columns = cols
    return temp

def groupby_min_pivot_Tool(trainDF, key, colname):
    temp = trainDF.groupby(key)\
            .apply(lambda df: df[colname].min())\
            .reset_index()\
            .rename(columns={0:'%s_min' % key[1]})\
            .pivot(columns=key[1], values='%s_min' % key[1], index=key[0])\
            .reset_index()
    cols = [temp.columns[0]] + ['%s_min' % key[1]+'_'+str(x) for x in temp.columns[1:]]
    temp.columns = cols
    return temp 

def groupby_std_pivot_Tool(trainDF, key, colname):
    temp = trainDF.groupby(key)\
            .apply(lambda df: df[colname].std())\
            .reset_index()\
            .rename(columns={0:'%s_std' % key[1]})\
            .pivot(columns=key[1], values='%s_std' % key[1], index=key[0])\
            .reset_index()
    cols = [temp.columns[0]] + ['%s_std' % key[1]+'_'+str(x) for x in temp.columns[1:]]
    temp.columns = cols
    return temp   


###########################train model functions####################################
#回归数据进行onthot
def ont_hotTool(trainDF, testDF, onthot_colnames):
    trainDF['split'] = 'train' 
    testDF['split'] = 'test' 
    DF = pd.concat([trainDF, testDF])
    DF = pd.get_dummies(DF, columns=onthot_colnames)
    trainDF = DF[DF['split'] == 'train'].drop('split', axis=1) 
    testDF = DF[DF['split'] == 'test'].drop('split', axis=1) 
    return trainDF, testDF



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    































