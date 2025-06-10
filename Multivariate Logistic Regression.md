import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tools import add_constant
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os

# 设置Matplotlib使用SimHei字体（或其他支持CJK的字体）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用于正常显示负号

# 定义保存路径
output_dir = r"D:/pythonproject/ardsbpri"  # 请根据需要修改为你的路径

# 1. 读取原始数据
data_path = r"D:/pythonproject/saaki/ARDSbpri.csv"
data = pd.read_csv(data_path)

# 2. 数据预处理

# 2.1 确保 'bpri' 列存在并生成 'bpri_log'
if 'bpri' in data.columns:
    # 避免 log(0) 错误，添加1
    data['bpri_log'] = np.log(data['bpri'] + 1)
else:
    raise ValueError("数据中缺少 'bpri' 列，无法生成 'bpri_log'")

# 2.2 排除指定变量
excluded_vars = ['early_aki', 'aki_stage', 'morality28d', 'hospital_expire_flag', 'icu_los']
data = data.drop(columns=excluded_vars, errors='ignore')  # 忽略不存在的列

# 2.3 定义显著变量

# **连续变量：**
continuous_vars = [
    'sofa_score', 'gcs', 'glucose', 'albumin', 'sapsii', 'avg_wbc',
    'avg_lymphocytes', 'bpri', 'pao2fio2ratio', 'cirrhosis',
    'bilirubin_total', 'sodium', 'bpri_log', 'calcium', 't', 'map',
    'base_excess', 'lactate', 'creatinine', 'ph', 'nmlr',
    'wbc', 'age', 'sii', 'height', 'potassium', 'rr', 'hr',
    'hb', 'bmi', 'charlson_comorbidity_index', 'avg_neutrophils_abs'
]

# **分类变量：**
categorical_vars = [
    'metastatic_solid_tumor', 'chronic_pulmonary_disease', 'paraplegia',
    'cerebrovascular_disease', 'hypertension', 'arrhythmia', 'malignant_cancer',
    'gender', 'race', 'peptic_ulcer_disease', 'aids', 'myocardial_infarct'
]

# **确保 'myocardial_infarct' 仅作为分类变量存在**
if 'myocardial_infarct' in continuous_vars:
    continuous_vars.remove('myocardial_infarct')

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

print("更新后的连续变量列表：", continuous_vars)
print("更新后的分类变量列表：", categorical_vars)

# 2.5 选择需要的列（显著变量 + 'ards' + 'bpri' + 'bpri_log'）
required_columns = continuous_vars + categorical_vars + ['ards', 'bpri', 'bpri_log']
# 移除重复的列
required_columns = list(dict.fromkeys(required_columns))
missing_required = set(required_columns) - set(data.columns)
if missing_required:
    raise ValueError(f"数据中缺少以下必要列：{missing_required}")

data = data[required_columns]

# 检查并移除重复列
data = data.loc[:, ~data.columns.duplicated()]

# 2.6 处理缺失值
# 确保因变量 'ards' 无缺失
data = data.dropna(subset=['ards'])

# 对于其他变量，选择删除包含缺失值的行
data = data.dropna()

# 2.7 编码分类变量
# 确保分类变量是 'category' 类型
for var in categorical_vars:
    data[var] = data[var].astype('category')

# 使用 pandas get_dummies 进行独热编码，避免虚拟变量陷阱（drop_first=True）
data_encoded = pd.get_dummies(data, columns=categorical_vars, drop_first=True)

# 2.8 转换布尔类型的虚拟变量为整数
bool_cols = data_encoded.select_dtypes(include=['bool']).columns
data_encoded[bool_cols] = data_encoded[bool_cols].astype(int)

# 2.9 准备自变量和因变量
X = data_encoded.drop(['ards'], axis=1)
y = data_encoded['ards']

# 2.10 确保所有自变量都是数值型
print("\n自变量的数据类型：")
print(X.dtypes)

# 检查是否存在非数值类型的列
non_numeric = X.select_dtypes(include=['object', 'category']).columns.tolist()
if non_numeric:
    print(f"警告：以下自变量为非数值类型，需要转换或移除：{non_numeric}")
    # 这里选择转换为数值类型
    X[non_numeric] = X[non_numeric].apply(pd.to_numeric, errors='coerce')
    # 再次检查是否有非数值类型
    non_numeric = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if non_numeric:
        raise ValueError(f"自变量仍包含非数值类型的列：{non_numeric}")
else:
    print("所有自变量均为数值型。")

# 2.11 添加常数项（截距）
X = sm.add_constant(X)

# 3. 构建并拟合多因素逻辑回归模型
try:
    model = sm.Logit(y, X)
    result = model.fit(disp=0)  # disp=0 关闭拟合过程的输出
    print("\n**模型拟合成功！**")
    print(result.summary())
except Exception as e:
    print(f"Error fitting the model: {e}")
    result = None

# “旧列名→新列名”映射
feature_name_map = {
    'hr': 'HR',
    'rr': 'RR',
    'sbp': 'SBP',
    'dbp': 'DBP',
    'map': 'MAP',
    'vis': 'VIS',
    'bpri_log': 'BPRI Log',
    'sapsii': 'SAPSII',
    'sofa_score': 'SOFA Score',
    'gcs': 'GCS',
    'age': 'Age',
    'icu_los': 'ICU Los',
    'wbc': 'WBC',
    'hematocrit': 'Hematocrit',
    'hb': 'Hb',
    'rbc': 'RBC',
    'avg_lymphocytes': 'lymphocytes abs',
    'avg_platelet': 'Platelet',
    'avg_monocytes_abs': 'monocytes abs',
    'avg_neutrophils_abs': 'neutrophils abs',
    'lymphocytes_%': 'Lymphocytes%',
    'monocytes_%': 'Monocytes%',
    'neutrophils_%': 'Neutrophils%',
    'nlr': 'nlr',
    'dnlr': 'dnlr',
    'mlr': 'mlr',
    'nmlr': 'nmlr',
    'siri': 'siri',
    'sii': 'sii',
    'bmi': 'BMI',
    'height': 'Height',
    'weight': 'Weight',
    'ph': 'PH',
    'pco2': 'PCO2',
    'po2': 'PO2',
    'pao2fio2ratio': 'Pao2Fio2ratio',
    'lactate': 'Lactate',
    'base_excess': 'Base Excess',
    'potassium': 'Potassium',
    'creatinine': 'Creatinine',
    'charlson_comorbidity_index': 'Charlson Comorbidity Index',
    't': 'T',
    'glucose': 'Glucose',
    'bilirubin_total': 'Total Bilirubin',
    'albumin': 'Albumin',
    'pt': 'PT',
    'ptt': 'PTT',
    'alt': 'ALT',
    'alp': 'ALP',
    'inr': 'INR',
    'ast': 'AST',
    'calcium': 'calcium',
    'sodium': 'sodium',
    'metastatic_solid_tumor': 'Metastatic Solid Tumor',
    'morality28d': 'Morality28d',
    'hospital_expire_flag': 'Hospital Expire Flag',
    'chronic_pulmonary_disease': 'Chronic Pulmonary Disease',
    'paraplegia': 'Paraplegia',
    'cerebrovascular_disease': 'Cerebrovascular Disease',
    'hypertension': 'Hypertension',
    'arrhythmia': 'Arrhythmia',
    'malignant_cancer': 'Malignant Cancer',
    'peptic_ulcer_disease': 'Peptic Ulcer Disease',
    'myocardial_infarct': 'Myocardial Infarct',
    'gender': 'Gender',
    'aids': 'Aids',
    'rheumatic_disease': 'Rheumatic Disease',
    'diabetes': 'Diabetes',
    'cirrhosis': 'Cirrhosis',
    'race_1.0': 'Race Black',
    'race_2.0': 'Race White',
    'race_3.0': 'Race Hispanic',
    'race_4.0': 'Race Asian',
    'race_5.0': 'Race Other'
}

# 4. 提取并显示BPRI和bpri_log的系数和P值
if result is not None:
    print("\n**BPRI指数 (bpri) 的多因素分析结果:**")
    if 'bpri' in result.params:
        bpri_coeff = result.params['bpri']
        bpri_pval = result.pvalues['bpri']
        bpri_or = np.exp(bpri_coeff)
        bpri_pval_formatted = f"<0.001" if bpri_pval < 0.001 else f"{bpri_pval:.3f}"

        # 使用 feature_name_map 替换变量名
        mapped_var_bpri = feature_name_map.get('bpri', 'bpri')
        print(f"{mapped_var_bpri}: OR = {bpri_or:.3f}, p值 = {bpri_pval_formatted}")
    else:
        print("BPRI指数 (bpri) 未包含在模型中.")

    print("\n**BPRI对数转换值 (bpri_log) 的多因素分析结果:**")
    if 'bpri_log' in result.params:
        bpri_log_coeff = result.params['bpri_log']
        bpri_log_pval = result.pvalues['bpri_log']
        bpri_log_or = np.exp(bpri_log_coeff)
        bpri_log_pval_formatted = f"<0.001" if bpri_log_pval < 0.001 else f"{bpri_log_pval:.3f}"

        # 使用 feature_name_map 替换变量名
        mapped_var_bpri_log = feature_name_map.get('bpri_log', 'bpri_log')
        print(f"{mapped_var_bpri_log}: OR = {bpri_log_or:.3f}, p值 = {bpri_log_pval_formatted}")
    else:
        print("BPRI对数转换值 (bpri_log) 未包含在模型中.")

# 5.3 提取显著变量并格式化P值，计算OR和95% CI
significant_results = result.pvalues[result.pvalues < 0.05].index.tolist()  # 提取显著变量
if 'const' in significant_results:
    significant_results.remove('const')  # 移除常数项

significant_vars = []
for var in significant_results:
    coef = result.params[var]
    std_err = result.bse[var]
    or_val = np.exp(coef)

    # 计算95%置信区间
    ci_lower = np.exp(coef - 1.96 * std_err)
    ci_upper = np.exp(coef + 1.96 * std_err)

    p_val = result.pvalues[var]
    p_val_formatted = f"<0.001" if p_val < 0.001 else f"{p_val:.3f}"

    # 使用 feature_name_map 替换变量名
    mapped_var = feature_name_map.get(var, var)

    significant_vars.append([mapped_var, f"{or_val:.3f}", f"{ci_lower:.3f}", f"{ci_upper:.3f}", p_val_formatted])
    print(f"{mapped_var}: OR = {or_val:.3f}, 95% CI = ({ci_lower:.3f}, {ci_upper:.3f}), p值 = {p_val_formatted}")

# 保存回归模型结果
if result is not None:
    model_summary = pd.DataFrame({
        'Variable': [feature_name_map.get(var, var) for var in result.params.index],
        'Coefficient': [f"{coef:.3f}" for coef in result.params],
        'Standard Error': [f"{se:.3f}" for se in result.bse],
        'OR': [f"{or_val:.3f}" for or_val in np.exp(result.params)],
        '95% CI Lower': [f"{ci_lower:.3f}" for ci_lower in np.exp(result.params - 1.96 * result.bse)],
        '95% CI Upper': [f"{ci_upper:.3f}" for ci_upper in np.exp(result.params + 1.96 * result.bse)],
        'P-Value': [f"{p_val:.3f}" for p_val in result.pvalues]
    })
    model_summary.to_csv(os.path.join(output_dir, 'logistic_regression_results_with_CI.csv'), index=False)

# 保存回归模型结果
if result is not None:
    model_summary = pd.DataFrame({
        'Variable': [feature_name_map.get(var, var) for var in result.params.index],
        'Coefficient': result.params,
        'P-Value': result.pvalues
    })
    model_summary.to_csv(os.path.join(output_dir, 'logistic_regression_results.csv'), index=False)
