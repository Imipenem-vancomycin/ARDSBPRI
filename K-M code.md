import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

# 定义保存路径
output_dir = r"D:/pythonproject/ardsbpri"  # 请根据需要修改为你的路径

# 1. 读取原始数据
data_path = r"D:/pythonproject/saaki/ARDSbpri.csv"
data = pd.read_csv(data_path)

# 2. 数据预处理

# 2.1 确保 'bpri' 列存在并生成 'bpri_log'
if 'bpri' in data.columns:
    data['bpri_log'] = np.log(data['bpri'] + 1)
else:
    raise ValueError("数据中缺少 'bpri' 列，无法生成 'bpri_log'")

# 2.2 排除指定变量
excluded_vars = ['early_aki', 'aki_stage', 'morality28d', 'hospital_expire_flag', ]
data = data.drop(columns=excluded_vars, errors='ignore')  # 忽略不存在的列

# 2.3 确保所有变量存在于数据中
continuous_vars = [
    'sofa_score', 'gcs', 'glucose', 'albumin', 'sapsii', 'avg_wbc',
    'avg_lymphocytes', 'cirrhosis', 'bilirubin_total', 'sodium', 'bpri_log', 'calcium',
    't', 'map', 'base_excess', 'lactate', 'creatinine', 'ph', 'nmlr',
    'wbc', 'age', 'sii', 'height', 'potassium', 'rr', 'hr', 'hb',
    'bmi', 'charlson_comorbidity_index', 'avg_neutrophils_abs'
]

categorical_vars = [
    'metastatic_solid_tumor', 'chronic_pulmonary_disease', 'paraplegia',
    'cerebrovascular_disease', 'hypertension', 'arrhythmia', 'malignant_cancer',
    'race', 'peptic_ulcer_disease', 'aids', 'myocardial_infarct', 'gender'
]

# 2.4 确保所有变量存在于数据中
missing_continuous = set(continuous_vars) - set(data.columns)
missing_categorical = set(categorical_vars) - set(data.columns)

if missing_continuous:
    print(f"警告：以下连续变量在数据中缺失，将被忽略：{missing_continuous}")

if missing_categorical:
    print(f"警告：以下分类变量在数据中缺失，将被忽略：{missing_categorical}")

# 更新变量列表，排除缺失的变量
continuous_vars = [var for var in continuous_vars if var in data.columns]
categorical_vars = [var for var in categorical_vars if var in data.columns]

# 2.5 选择需要的列（显著变量 + 'ards' + 'bpri_log' + 'icu_los'）
required_columns = continuous_vars + categorical_vars + ['ards', 'bpri_log', 'icu_los']
required_columns = list(dict.fromkeys(required_columns))
missing_required = set(required_columns) - set(data.columns)
if missing_required:
    raise ValueError(f"数据中缺少以下必要列：{missing_required}")

data = data[required_columns]

# 2.6 处理缺失值
data = data.dropna(subset=['ards'])
data = data.dropna()

# 2.7 编码分类变量
for var in categorical_vars:
    data[var] = data[var].astype('category')

data_encoded = pd.get_dummies(data, columns=categorical_vars, drop_first=True)

# 2.8 转换布尔类型的虚拟变量为整数
bool_cols = data_encoded.select_dtypes(include=['bool']).columns
data_encoded[bool_cols] = data_encoded[bool_cols].astype(int)

# 2.9 对 'bpri_log' 进行分组（四分位数分组）
data_encoded['bpri_log_group'] = pd.qcut(data_encoded['bpri_log'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

# 3. Kaplan-Meier分析
kmf = KaplanMeierFitter()

# 3.1 画出不同bpri_log组的Kaplan-Meier曲线
plt.figure(figsize=(10, 6))

for group in data_encoded['bpri_log_group'].unique():
    # 提取bpri_log分组数据
    group_data = data_encoded[data_encoded['bpri_log_group'] == group]
    # 事件列：ARDS发生与否；时间列：ICU住院天数
    kmf.fit(group_data['icu_los'], event_observed=group_data['ards'], label=group)
    kmf.plot_survival_function()

plt.title("Kaplan-Meier Curve by bpri_log Group")
plt.xlabel("ICU Stay Days")
plt.ylabel("Cumulative ARDS Incidence")
plt.legend(title="bpri_log Group")
plt.show()

# 4. Log-rank检验：比较不同bpri_log组之间的生存曲线差异
# 进行Log-rank检验，检查不同组之间的差异是否显著
results = {}

for group1 in data_encoded['bpri_log_group'].unique():
    for group2 in data_encoded['bpri_log_group'].unique():
        if group1 < group2:  # 每对组合只比较一次
            group1_data = data_encoded[data_encoded['bpri_log_group'] == group1]
            group2_data = data_encoded[data_encoded['bpri_log_group'] == group2]

            # Log-rank检验
            results[(group1, group2)] = logrank_test(group1_data['icu_los'], group2_data['icu_los'],
                                                     event_observed_A=group1_data['ards'],
                                                     event_observed_B=group2_data['ards'])

# 显示Log-rank检验结果
for group_pair, result in results.items():
    print(f"Log-rank test between {group_pair[0]} and {group_pair[1]}: p-value = {result.p_value:.4f}")
