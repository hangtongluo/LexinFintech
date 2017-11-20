# -- coding: utf-8 --
import pandas as pd
import numpy as np

#读取训练集文件
ud_mdl = pd.read_csv(r"data\lexin_train\ud_mdl.csv")

#读取标签，提取标签，并且将训练集与标签连接
dep_mdl = pd.read_csv(r'data\lexin_train\dep_mdl.csv')
dep_mdl = dep_mdl[['fuid_md5','dep']]
ud_mdl = pd.merge(ud_mdl,dep_mdl,how='left',on='fuid_md5')

#读取预测文件
ud_offtime = pd.read_csv(r"data\lexin_test\ud_offtime.csv")

#将预测与训练根据列连接，一共100000行数据
ud_mdl = pd.concat([ud_mdl,ud_offtime],axis=0).reset_index()
ud_mdl = ud_mdl.drop('index',axis=1)

#籍贯省份我打算用学校省份填充
ud_mdl['fdomicile_provice'] = ud_mdl['fdomicile_provice'].fillna(ud_mdl['sch_fprovince_name'])

#学校省份我也用籍贯省份填充
ud_mdl['sch_fprovince_name'] = ud_mdl['sch_fprovince_name'].fillna(ud_mdl['fdomicile_provice'])

#剩下的一些城市的缺失值我直接用缺失值填充的，因为确实不好用其他填充，所以只能把缺失作为一种特征
ud_mdl['fdomicile_city'] = ud_mdl['fdomicile_city'].fillna('籍贯城市_缺失')
ud_mdl['sch_fcity_name'] = ud_mdl['sch_fcity_name'].fillna('学校城市_缺失')
ud_mdl['fdomicile_area'] = ud_mdl['fdomicile_area'].fillna('籍贯县市_缺失')
ud_mdl['sch_fregion_name'] = ud_mdl['sch_fregion_name'].fillna('学校县市_缺失')
ud_mdl['sch_fcompany_name'] = ud_mdl['sch_fcompany_name'].fillna('学校片区名_缺失')

#注册时间转换，太笨了，用了这么多行程序
ud_mdl['fregister_time_day']=ud_mdl['fregister_time'].str.slice(0,2)
ud_mdl['fregister_time_month']=ud_mdl['fregister_time'].str.slice(2,5)
ud_mdl['fregister_time_year']=ud_mdl['fregister_time'].str.slice(5,7)
ud_mdl['fregister_time_month']=ud_mdl['fregister_time_month'].replace(['SEP','OCT','JUN','AUG','JUL','MAY','APR','MAR','DEC','NOV','JAN','FEB'],[9,10,6,8,7,5,4,3,12,11,1,2])
ud_mdl['fregister_time_year']=str(20)+ud_mdl['fregister_time_year']

ud_mdl['fregister_time'] = ud_mdl['fregister_time_year'].apply(lambda x:str(x))+'-'+ud_mdl['fregister_time_month'].apply(lambda x:str(x))
ud_mdl['fregister_time'] = ud_mdl['fregister_time'].apply(lambda x:str(x))+'-'+ud_mdl['fregister_time_day'].apply(lambda x:str(x))
from datetime import datetime
ud_mdl['fregister_time'] = ud_mdl['fregister_time'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
ud_mdl=ud_mdl.drop(['fregister_time_day','fregister_time_month','fregister_time_year'],axis=1)
#ud_mdl=ud_mdl.drop(['fregister_time_month','fregister_time_year'],axis=1)


#审核通过时间
ud_mdl['fpocket_auth_time_day']=ud_mdl['fpocket_auth_time'].str.slice(0,2)
ud_mdl['fpocket_auth_time_month']=ud_mdl['fpocket_auth_time'].str.slice(2,5)
ud_mdl['fpocket_auth_time_year']=ud_mdl['fpocket_auth_time'].str.slice(5,7)
ud_mdl['fpocket_auth_time_month']=ud_mdl['fpocket_auth_time_month'].replace(['SEP','OCT','JUN','AUG','JUL','MAY','APR','MAR','DEC','NOV','JAN','FEB'],[9,10,6,8,7,5,4,3,12,11,1,2])
ud_mdl['fpocket_auth_time_year']=str(20)+ud_mdl['fpocket_auth_time_year']

ud_mdl['fpocket_auth_time'] = ud_mdl['fpocket_auth_time_year'].apply(lambda x:str(x))+'-'+ud_mdl['fpocket_auth_time_month'].apply(lambda x:str(x))
ud_mdl['fpocket_auth_time'] = ud_mdl['fpocket_auth_time'].apply(lambda x:str(x))+'-'+ud_mdl['fpocket_auth_time_day'].apply(lambda x:str(x))
from datetime import datetime
ud_mdl['fpocket_auth_time'] = ud_mdl['fpocket_auth_time'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
#ud_mdl=ud_mdl.drop(['fpocket_auth_time_day','fpocket_auth_time_month','fpocket_auth_time_year'],axis=1)
ud_mdl=ud_mdl.drop(['fpocket_auth_time_day'],axis=1)

#毕业时间填充
ud_mdl['fcal_graduation_day']=ud_mdl['fcal_graduation'].str.slice(0,2)
ud_mdl['fcal_graduation_month']=ud_mdl['fcal_graduation'].str.slice(2,5)
ud_mdl['fcal_graduation_year']=ud_mdl['fcal_graduation'].str.slice(5,7)

ud_mdl['fcal_graduation_day'] = ud_mdl['fcal_graduation_day'].fillna(str(1))
ud_mdl['fcal_graduation_month'] = ud_mdl['fcal_graduation_month'].fillna(str(7))
ud_mdl['fcal_graduation_year'] = ud_mdl['fcal_graduation_year'].fillna(str(17))

ud_mdl['fcal_graduation_month']=ud_mdl['fcal_graduation_month'].replace(['JUL'],[7])
ud_mdl['fcal_graduation_year']=str(20)+ud_mdl['fcal_graduation_year']
ud_mdl['fcal_graduation'] = ud_mdl['fcal_graduation_year'].apply(lambda x:str(x))+'-'+ud_mdl['fcal_graduation_month'].apply(lambda x:str(x))
ud_mdl['fcal_graduation'] = ud_mdl['fcal_graduation'].apply(lambda x:str(x))+'-'+ud_mdl['fcal_graduation_day'].apply(lambda x:str(x))

from datetime import datetime
ud_mdl['fcal_graduation'] = ud_mdl['fcal_graduation'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))
#ud_mdl = ud_mdl.drop(['fcal_graduation_day','fcal_graduation_month','fcal_graduation_year'],axis=1)
ud_mdl = ud_mdl.drop(['fcal_graduation_day','fcal_graduation_month'],axis=1)

#授信时间减去注册时间fpocket_fregister,我感觉这是一个比较好的特征，有待验证
ud_mdl['fpocket_fregister'] = ud_mdl['fpocket_auth_time'] - ud_mdl['fregister_time']
ud_mdl['fpocket_fregister'] = ud_mdl['fpocket_fregister'].apply(lambda x:str(x).split(' ')[0])

ud_mdl['fpocket_fregister'] = ud_mdl['fpocket_fregister'].apply(lambda x:int(x))
ud_mdl['fpocket_fregister'] = np.log(ud_mdl['fpocket_fregister']+2)

ud_mdl['fpocket_fregister'] = ud_mdl['fpocket_fregister'].apply(lambda x:int(x))

#毕业时间减去授信时间，好像时间有负的时间
ud_mdl['fcal_fpocket'] = ud_mdl['fcal_graduation'] - ud_mdl['fpocket_auth_time']
ud_mdl['fcal_fpocket'] = ud_mdl['fcal_fpocket'].apply(lambda x:str(x).split(' ')[0])
#毕业时间减去注册时间
ud_mdl['fcal_fregister'] = ud_mdl['fcal_graduation'] - ud_mdl['fregister_time']
ud_mdl['fcal_fregister'] = ud_mdl['fcal_fregister'].apply(lambda x:str(x).split(' ')[0])

#将学校名称进行计数，然后替换学校名称
fschoolarea_name_md5 = ud_mdl.fschoolarea_name_md5.value_counts().reset_index()
fschoolarea_name_md5.rename(columns={'index': 'fschoolarea_name_md5', 'fschoolarea_name_md5': 'fschoolarea_name_md5_counts'}, inplace=True)
ud_mdl = pd.merge(ud_mdl,fschoolarea_name_md5,how='left',on='fschoolarea_name_md5')
ud_mdl=ud_mdl.drop('fschoolarea_name_md5',axis=1)

#将学校个数log后再进行离散化（？？？？？？？？？？？？？？？？？？？？？？？？？？）
ud_mdl['fschoolarea_name_md5_counts'] = np.log(ud_mdl['fschoolarea_name_md5_counts'])
ud_mdl['fschoolarea_name_md5_counts']=ud_mdl['fschoolarea_name_md5_counts'].apply(lambda x:int(x))

#学校人数处理
nnn = ud_mdl['fstd_num'].value_counts().reset_index()
nnn.columns = ['fstd_num','fstd_num_counts']
ud_mdl = pd.merge(ud_mdl,nnn,how='left',on='fstd_num')
#ud_mdl=ud_mdl.drop('fstd_num',axis=1)

ud_mdl['fstd_num_counts'] = ud_mdl['fstd_num_counts'].apply(lambda x:int(np.log(x+1)))
ud_mdl['fstd_num'] = ud_mdl['fstd_num'].apply(lambda x:int(np.log(x+1)))

#籍贯城市处理
ooo = ud_mdl['fdomicile_city'].value_counts().reset_index()
ooo.columns = ['fdomicile_city','fdomicile_city_counts']
ud_mdl = pd.merge(ud_mdl,ooo,how='left',on='fdomicile_city')
#ud_mdl = ud_mdl.drop('fdomicile_city',axis=1)

ud_mdl['fdomicile_city_counts'] = ud_mdl['fdomicile_city_counts'].apply(lambda x:int(np.log(x)))

#学校城市处理
ppp = ud_mdl['sch_fcity_name'].value_counts().reset_index()
ppp.columns = ['sch_fcity_name','sch_fcity_name_counts']
ud_mdl = pd.merge(ud_mdl,ppp,how='left',on='sch_fcity_name')
#ud_mdl = ud_mdl.drop('sch_fcity_name',axis=1)

ud_mdl['sch_fcity_name_counts'] = ud_mdl['sch_fcity_name_counts'].apply(lambda x:int(np.log(x)))

#去掉时间两列，标签，以及id
ud_mdl = ud_mdl.drop(['fregister_time','fpocket_auth_time'],axis=1)


#标签编码
from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
ud_mdl['fdomicile_area'] = class_le.fit_transform(ud_mdl['fdomicile_area'].values)

class_le = LabelEncoder()
ud_mdl['sch_fcompany_name'] = class_le.fit_transform(ud_mdl['sch_fcompany_name'].values)

class_le = LabelEncoder()
ud_mdl['sch_fregion_name'] = class_le.fit_transform(ud_mdl['sch_fregion_name'].values)

class_le = LabelEncoder()
ud_mdl['fdomicile_city'] = class_le.fit_transform(ud_mdl['fdomicile_city'].values)

class_le = LabelEncoder()
ud_mdl['sch_fcity_name'] = class_le.fit_transform(ud_mdl['sch_fcity_name'].values)

#ud_mdl = ud_mdl.drop('fcal_graduation',axis=1)

#独热编码
sch_fprovince_name=pd.get_dummies(ud_mdl['sch_fprovince_name'])
ud_mdl = pd.concat([ud_mdl,sch_fprovince_name],axis=1)
fdomicile_provice=pd.get_dummies(ud_mdl['fdomicile_provice'])
ud_mdl = pd.concat([ud_mdl,fdomicile_provice],axis=1)
ud_mdl = ud_mdl.drop(['fdomicile_provice','sch_fprovince_name'],axis=1)

fage=pd.get_dummies(ud_mdl['fage'])
ud_mdl = pd.concat([ud_mdl,fage],axis=1)
fauth_source_type=pd.get_dummies(ud_mdl['fauth_source_type'])
ud_mdl = pd.concat([ud_mdl,fauth_source_type],axis=1)
ud_mdl = ud_mdl.drop(['fage','fauth_source_type'],axis=1)

fcal_graduation=pd.get_dummies(ud_mdl['fcal_graduation'])
ud_mdl = pd.concat([ud_mdl,fcal_graduation],axis=1)
fcollege_level=pd.get_dummies(ud_mdl['fcollege_level'])
ud_mdl = pd.concat([ud_mdl,fcollege_level],axis=1)
ud_mdl = ud_mdl.drop(['fcal_graduation','fcollege_level'],axis=1)

fsex=pd.get_dummies(ud_mdl['fsex'])
ud_mdl = pd.concat([ud_mdl,fsex],axis=1)
fstd_num=pd.get_dummies(ud_mdl['fstd_num'])
ud_mdl = pd.concat([ud_mdl,fstd_num],axis=1)
ud_mdl = ud_mdl.drop(['fsex','fstd_num'],axis=1)

fis_entrance_exam=pd.get_dummies(ud_mdl['fis_entrance_exam'])
ud_mdl = pd.concat([ud_mdl,fis_entrance_exam],axis=1)
fpocket_fregister=pd.get_dummies(ud_mdl['fpocket_fregister'])
ud_mdl = pd.concat([ud_mdl,fpocket_fregister],axis=1)
ud_mdl = ud_mdl.drop(['fis_entrance_exam','fpocket_fregister'],axis=1)

fschoolarea_name_md5_counts=pd.get_dummies(ud_mdl['fschoolarea_name_md5_counts'])
ud_mdl = pd.concat([ud_mdl,fschoolarea_name_md5_counts],axis=1)
fdomicile_city_counts=pd.get_dummies(ud_mdl['fdomicile_city_counts'])
ud_mdl = pd.concat([ud_mdl,fdomicile_city_counts],axis=1)
ud_mdl = ud_mdl.drop(['fschoolarea_name_md5_counts','fdomicile_city_counts'],axis=1)

sch_fcity_name_counts=pd.get_dummies(ud_mdl['sch_fcity_name_counts'])
ud_mdl = pd.concat([ud_mdl,sch_fcity_name_counts],axis=1)

ud_mdl = ud_mdl.drop(['sch_fcity_name_counts'],axis=1)

ud_mdl.head()
ud_train = ud_mdl[0:50000]
ud_test = ud_mdl[50000:100000]
ud_train.to_csv(r'features/ud_train.csv',index=False,encoding='gb2312')
ud_test.to_csv(r'features/ud_test.csv',index=False,encoding='gb2312')