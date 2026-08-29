import numpy as np
import pandas as pd

# 模型
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 评价
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


#导入数据，low_memory=False 避免数据类型推断错误
data = pd.read_csv(
    "data/raw/train_transaction.csv",
    usecols=[
        "TransactionDT",
        "TransactionAmt",
        "ProductCD",
        "card4",
        "card6",
        "P_emaildomain",
        "R_emaildomain",
        "isFraud",
        "card1",
        "card2",
        "card5",
        "addr1",
        "addr2",
        "C1", "C2", "C3", "C4", "C5", "C6", "C7","C8", "C9", "C10", "C11", "C12", "C13", "C14",
        "D1", "D2", "D3", "D4", "D5","D6", "D7", "D8", "D9", "D10","D11", "D12", "D13", "D14", "D15",
    ],
    low_memory=False
)

# 将 TransactionDT 转换为小时和天数以便训练模型，TransactionAmt 取对数便于计算（减少极端值影响）
data['Transaction_Hour'] = (data['TransactionDT'] // 3600) % 24
data['Transaction_Day'] = data['TransactionDT'] // 86400
data['Transaction_Amt_Log'] = np.log1p(data['TransactionAmt'])

# 根据时间顺序分割出训练集与验证集
cutoff = data['TransactionDT'].quantile(0.8)

train_part = data[data['TransactionDT'] <= cutoff].copy()
valid_part = data[data['TransactionDT'] > cutoff].copy()

# 分离标签
y_train = train_part['isFraud']
y_valid = valid_part['isFraud']

# 训练集中正常交易约是欺诈交易的 约27 倍。
negative_count = (y_train == 0).sum() 
positive_count = (y_train == 1).sum() # 训练集中欺诈总样本数

scale_pos_weight = negative_count / positive_count

print("scale_pos_weight:", scale_pos_weight)


# 生成频次特征，1列替代原始高基数列，避免OneHot维度爆炸(由category变成numeric的频率，不走onehot，对模型来说频率有用)

#高基数类别字段（不能使用onehot）
high_cardinality_features = [
    "card1",
    "card2",
    "card5",
    "addr1",
    "addr2",
]

frequency_features = []

for column in high_cardinality_features:
    train_values = train_part[column].fillna('missing')
    valid_values = valid_part[column].fillna('missing')

    frequency_map = train_values.value_counts(normalize=True)

    feature_name = f'{column}_frequency'

    train_part[feature_name] = train_values.map(frequency_map)
    valid_part[feature_name] = valid_values.map(frequency_map).fillna(0)

    frequency_features.append(feature_name)

    print(
        column,
        "验证集中未见类别比例：",
        (valid_part[feature_name] == 0).mean()
    )

# 加入C组做消融实验
C_features = [
    "C1", "C2", "C3", "C4", "C5", "C6", "C7",
    "C8", "C9", "C10", "C11", "C12", "C13", "C14",
]

#加入D组做消融实验
D_features = [
    "D1", "D2", "D3", "D4", "D5",
    "D6", "D7", "D8", "D9", "D10",
    "D11", "D12", "D13", "D14", "D15",
]

D_missing_features = []

for column in D_features:

    feature_name = f"{column}_missing"

    train_part[feature_name] = (train_part[column].isna().astype('int8'))
    valid_part[feature_name] = (valid_part[column].isna().astype('int8'))

    D_missing_features.append(feature_name)


# 定义数值特征和类别特征
numeric_features = [
    "TransactionAmt",
    "Transaction_Amt_Log",
    "Transaction_Hour",
    "Transaction_Day",
] + frequency_features + C_features + D_features + D_missing_features

categorical_features = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
]

x_train = train_part[numeric_features + categorical_features]
x_valid = valid_part[numeric_features + categorical_features]



# 创建pipeline，数值特征使用中位数填充缺失值并标准化，类别特征使用常量填充缺失值并进行独热编码
numeric_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy='median')),#所有缺失值都填充为【训练集计算得到的】中位数
    ('scaler',StandardScaler())#所有数值等比转换为：平均值等于0，方差等于1
])

categorical_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy='constant',fill_value='missing')), #所有缺失值都填充为常量'missing'，缺失变成一个独立类别
    ('onehot',OneHotEncoder(handle_unknown='ignore')) #填ignore遇到新类别不报错（默认error报错）
    # `OneHotEncoder`：独热编码，分类特征转 0‑1 稀疏矩阵，造成特征膨胀。city -> city_北京,city_上海,city_深圳（在北京就写[1,0,0])
])

# 创建 ColumnTransformer，将数值特征和类别特征分别应用不同的pipeline
preprocessor = ColumnTransformer([
    ('numeric',numeric_pipeline,numeric_features), #"步骤名称"，调用pipeline，特征列表
    ('categorical',categorical_pipeline,categorical_features)
])

# 创建完整的pipeline,训练模型参数
model_pipeline = Pipeline([
    ("preprocessor", preprocessor), # 预处理器
    ("model",XGBClassifier(  #决策树模型
    objective = 'binary:logistic', # 标注二分类任务
    eval_metric = 'aucpr', # 以pr-auc为指标，后面可用于输出日志和评判早停
    n_estimators = 414, # 最大可以训练300棵树（不考虑早停情况下）
    max_depth = 6,   # 每棵树最多分裂6次
    learning_rate = 0.05, # 每棵树的学习步长，越小越稳但需要更多树(配合n_estimators综合考量 少量多次)
    subsample = 0.8,   # 每次随机抽样80%用于训练               
    colsample_bytree = 0.8,  # 每次抽取80%的特征用于训练
    min_child_weight= 10, # 限制节点最小有效权重，避免过拟合
    reg_lambda = 1, # 抑制分裂 避免极端分数
    scale_pos_weight = scale_pos_weight, #应对样本不平衡（scale_pos_weight = negative_count / positive_count提前算好）
    tree_method = 'hist', #采用直方图分割，相比‘exact’参数加速训练
    random_state = 42,
    n_jobs = -1
    ))
])

# 训练模型
model_pipeline.fit(x_train,y_train) #


# 预测验证集的欺诈概率
#predict_proba返回每个类别的概率，[:,1]表示取正类（欺诈）的概率(predict直接给出概率更大的预测结果，无原始概率数据)
valid_probabilities = model_pipeline.predict_proba(x_valid)[:, 1] 

# 模型评价
# 预测验证集的欺诈标签，阈值为0.5
valid_predictions = (valid_probabilities >= 0.5).astype(int)

roc_auc = roc_auc_score(y_valid,valid_probabilities) 
pr_auc = average_precision_score(y_valid,valid_probabilities) #针对正负分布不平均更有价值
precision = precision_score(y_valid,valid_predictions,zero_division=0)
recall = recall_score(y_valid,valid_predictions,zero_division=0)
f1 = f1_score(y_valid, valid_predictions, zero_division=0) #综合precision和recall，两个都高f1才高
matrix = confusion_matrix(y_valid, valid_predictions) # [[TN FP]
                                                      #  [FN TP]]   这里‘欺诈’是1 positive，‘正常’是0 negative

print("\n模型评价：")
print("ROC-AUC:", roc_auc)
print("PR-AUC:", pr_auc)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)

print("\n混淆矩阵：")
print(matrix)

# 第一次结果（默认阈值0.5）不理想误报较多，因此比较不同阈值的风险预警取舍
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
rows = []
for threshold in thresholds:
    predictions = (valid_probabilities >= threshold).astype(int)

    rows.append({
        'threshold':threshold,
        'alerts':predictions.sum(),
        'alert_rate':predictions.mean(),
        'precision':precision_score(y_valid,predictions,zero_division=0),
        'recall':recall_score(y_valid,predictions,zero_division=0),
        'f1':f1_score(y_valid,predictions,zero_division=0)
    })

threshold_results = pd.DataFrame(rows)

print("\n不同阈值的结果：")
print(threshold_results)

# 按照欺诈概率排序
ranking = pd.DataFrame({
    'isFraud':y_valid.to_numpy(),
    "fraud_probability": valid_probabilities
}).sort_values('fraud_probability',ascending = False)


# 如果风控团队每天只能人工复核风险最高的前 1%、前 5%、前 10% 交易，能抓住多少欺诈？
for top_rate in [0.01,0.05,0.10]:
    alert_count = int(len(ranking) * top_rate)
    top_alerts = ranking.head(alert_count)

    captured_fraud = top_alerts["isFraud"].sum()
    total_fraud = ranking["isFraud"].sum()

    print(f"\n审核前 {top_rate:.0%} 的交易：")
    print("报警数量：", alert_count)
    print("抓住的欺诈数：", captured_fraud)
    print("欺诈捕获率：", captured_fraud / total_fraud)
    print("报警中的欺诈率：", top_alerts["isFraud"].mean())


