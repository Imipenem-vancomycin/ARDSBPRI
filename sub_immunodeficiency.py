import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
from statsmodels.stats.outliers_influence import variance_inflation_factor

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
excluded_vars = ['early_aki', 'aki_stage', 'morality28d', 'hospital_expire_flag', 'icu_los']
data = data.drop(columns=excluded_vars, errors='ignore')  # 忽略不存在的列

# 2.3 定义显著变量
continuous_vars = [
    'sofa_score', 'gcs', 'glucose', 'albumin', 'sapsii', 'avg_wbc',
    'avg_lymphocytes', 'pao2fio2ratio', 'cirrhosis',
    'bilirubin_total', 'sodium', 'bpri_log', 'calcium', 't', 'map',
    'base_excess', 'lactate', 'creatinine', 'ph', 'nmlr',
    'wbc', 'age', 'sii', 'height', 'potassium', 'rr', 'hr',
    'hb', 'bmi', 'charlson_comorbidity_index', 'avg_neutrophils_abs'
]

categorical_vars = [
    'metastatic_solid_tumor', 'chronic_pulmonary_disease', 'paraplegia',
    'cerebrovascular_disease', 'hypertension', 'arrhythmia', 'malignant_cancer',
    'race', 'peptic_ulcer_disease', 'aids', 'myocardial_infarct'
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

# 2.5 选择需要的列（显著变量 + 'ards' + 'bpri_log'）
required_columns = continuous_vars + categorical_vars + ['ards', 'bpri_log']
required_columns = list(dict.fromkeys(required_columns))  # 去除重复列
missing_required = set(required_columns) - set(data.columns)
if missing_required:
    raise ValueError(f"数据中缺少以下必要列：{missing_required}")

data = data[required_columns]

# 2.6 处理缺失值
data = data.dropna(subset=['ards'])  # 删除 'ards' 缺失的行
data = data.dropna()  # 删除任何缺失值行

# 2.7 编码分类变量
for var in categorical_vars:
    data[var] = data[var].astype('category')

data_encoded = pd.get_dummies(data, columns=categorical_vars, drop_first=True)

# 2.8 转换布尔类型的虚拟变量为整数
bool_cols = data_encoded.select_dtypes(include=['bool']).columns
data_encoded[bool_cols] = data_encoded[bool_cols].astype(int)

# 2.9 打印 data_encoded 的列名，确认是否有必要的列
print("data_encoded 的列名：")
print(data_encoded.columns.tolist())

# 3. 定义免疫缺陷组

# 3.1 定义免疫缺陷组
# 假设分类变量已经被独热编码，如 'malignant_cancer_1.0', 'metastatic_solid_tumor_1.0', 'aids_1.0'
data_encoded['immunodeficiency'] = np.where(
    (data_encoded.get('malignant_cancer_1.0', 0) == 1) |
    (data_encoded.get('metastatic_solid_tumor_1.0', 0) == 1) |
    (data_encoded.get('aids_1.0', 0) == 1),
    1,
    0
)

# 3.2 确保 'immunodeficiency' 列被正确创建
print("\nimmunodeficiency 列的值分布：")
print(data_encoded['immunodeficiency'].value_counts())

# 3.3 排除用于定义免疫缺陷组的变量（避免变量泄漏）
vars_to_exclude = ['malignant_cancer_1.0', 'metastatic_solid_tumor_1.0', 'aids_1.0']
data_encoded = data_encoded.drop(columns=vars_to_exclude, errors='ignore')

# 3.4 再次打印 data_encoded 的列名，确认变量是否被移除
print("\ndata_encoded 排除定义组别变量后的列名：")
print(data_encoded.columns.tolist())

# 4. 分组分析：免疫缺陷 vs 非免疫缺陷

# 4.1 根据 'immunodeficiency' 列进行分组
data_imm_def = data_encoded[data_encoded['immunodeficiency'] == 1]  # 免疫缺陷组
data_non_imm_def = data_encoded[data_encoded['immunodeficiency'] == 0]  # 非免疫缺陷组

# 4.2 准备自变量和因变量（免疫缺陷组和非免疫缺陷组）
X_imm_def = data_imm_def.drop(['ards', 'immunodeficiency'], axis=1)
y_imm_def = data_imm_def['ards']

X_non_imm_def = data_non_imm_def.drop(['ards', 'immunodeficiency'], axis=1)
y_non_imm_def = data_non_imm_def['ards']

# 4.3 添加常数项（截距）
X_imm_def = sm.add_constant(X_imm_def)
X_non_imm_def = sm.add_constant(X_non_imm_def)

# 5. 拟合逻辑回归模型
def fit_logistic_model(X, y):
    try:
        model = sm.Logit(y, X)
        result = model.fit(disp=0)
        return result
    except Exception as e:
        print(f"模型拟合错误：{e}")
        return None

result_imm_def = fit_logistic_model(X_imm_def, y_imm_def)
result_non_imm_def = fit_logistic_model(X_non_imm_def, y_non_imm_def)

# 6. 显示和保存结果
def print_model_results(result, group_name):
    if result is not None:
        print(f"\n**{group_name}组的多因素分析结果:**")
        print(result.summary())

        # 提取bpri_log的OR和P值
        if 'bpri_log' in result.params:
            coef = result.params['bpri_log']
            pval = result.pvalues['bpri_log']
            or_val = np.exp(coef)
            print(f"bpri_log: OR = {or_val:.3f}, p值 = {pval:.3f}")
        else:
            print("模型中未包含 'bpri_log' 变量。")
    else:
        print(f"{group_name}组的模型未成功拟合.")

def save_model_results(result, group_name):
    if result is not None:
        result_df = result.summary2().tables[1]
        result_df.to_csv(os.path.join(output_dir, f"{group_name}_model_results.csv"))

# 打印和保存免疫缺陷组和非免疫缺陷组的结果
print_model_results(result_imm_def, "免疫缺陷")
print_model_results(result_non_imm_def, "非免疫缺陷")

save_model_results(result_imm_def, "免疫缺陷")
save_model_results(result_non_imm_def, "非免疫缺陷")

# 7. 比较两个组之间的 'bpri_log' 的OR是否存在显著差异

# 7.1 创建交互项：bpri_log * immunodeficiency
data_encoded['bpri_log_immunodeficiency'] = data_encoded['bpri_log'] * data_encoded['immunodeficiency']

# 7.2 准备自变量，包括交互项
# 选择所有自变量，包括新的交互项
# 确保 'immunodeficiency' 和 'bpri_log_immunodeficiency' 包含在自变量中
X = data_encoded.drop(['ards'], axis=1)  # 因变量已在 'y' 中定义

# 添加常数项
X = sm.add_constant(X)

# 7.3 定义因变量
y = data_encoded['ards']

# 7.4 拟合包含交互项的逻辑回归模型
model_interaction = sm.Logit(y, X)
result_interaction = model_interaction.fit()

# 7.5 输出回归结果
print("\n**包含交互项的逻辑回归模型结果:**")
print(result_interaction.summary())

# 7.6 提取并显示 bpri_log 和交互项的相关信息
print(f"\nbpri_log 的回归系数： {result_interaction.params['bpri_log']}, p值： {result_interaction.pvalues['bpri_log']}")
print(f"bpri_log_immunodeficiency 的回归系数： {result_interaction.params['bpri_log_immunodeficiency']}, p值： {result_interaction.pvalues['bpri_log_immunodeficiency']}")

# 7.7 解释结果
if result_interaction.pvalues['bpri_log_immunodeficiency'] < 0.05:
    print("\nbpri_log 的 OR 在免疫缺陷组和非免疫缺陷组之间存在显著差异。")
else:
    print("\nbpri_log 的 OR 在免疫缺陷组和非免疫缺陷组之间不存在显著差异。")

# 8. 计算并打印VIF
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data['feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

print("\n免疫缺陷组的VIF:")
vif_imm_def = calculate_vif(X_imm_def.drop('const', axis=1))  # 去除常数项
print(vif_imm_def)

# 如果非免疫缺陷组的模型成功拟合，则计算其VIF
if result_non_imm_def is not None:
    print("\n非免疫缺陷组的VIF:")
    vif_non_imm_def = calculate_vif(X_non_imm_def.drop('const', axis=1))
    print(vif_non_imm_def)
else:
    print("\n非免疫缺陷组的VIF无法计算，因为模型未成功拟合。")
