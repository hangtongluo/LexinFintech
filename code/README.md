
主要代码：  
utils.py：一些需要的工具函数  
basis_pro.py：前期数据的处理  
features_extract.py：提取特征数据  
ud_mdl.py：用户基本信息提取  
p6M_mdl.py：用户过去6个月信息提取  
login_scene_mdl.py：用户过去6个月场景浏览统计  
  
lgb_model_auc1.py：auc预测lgb模型1  
lgb_model_auc2.py：auc预测lgb模型2  
lgb_model_re.py：回归预测lgb模型  
xgb_model_auc1.py：auc预测xgb模型1  

sub_auc.py：auc提交文件生成  
sub_re1.py：回归提交文件生成1  
sub_re2.py：回归提交文件生成2  

sub_ronghe_auc.py：auc融合文件生成  
sub_ronghe_re.py：回归融合文件生成  

文件夹：  
data：训练和测试原始数据
	---- lexin_test 存放测试数据
	---- lexin_train 存放训练数据
features：保存提取的特征
model：存放模型和特征重要性排序文件
pro_data：保存中间预处理数据
submission：保存提交结果
	---- xgb 存放xgb模型结果
	---- lgb 存放lgb模型结果
	---- combin 存放模型融合结果

v_3_4_最差.R：模型融合使用
v_5_10_0.924.R：模型融合使用
