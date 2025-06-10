import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

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

# 2.5 选择需要的列（显著变量 + 'ards' + 'bpri_log' + 'pao2fio2ratio'）
required_columns = continuous_vars + categorical_vars + ['ards', 'bpri_log', 'pao2fio2ratio']
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

# 2.9 对sofa_score进行分组（0-6为一组，7以上为一组）
data_encoded['sofa_score_group'] = pd.cut(
    data_encoded['sofa_score'],
    bins=[-np.inf, 6, np.inf],
    labels=['0-6', '7+']
)

# 2.10 对每个亚组进行多因素分析
groups = data_encoded['sofa_score_group'].unique()

for group in groups:
    subgroup_data = data_encoded[data_encoded['sofa_score_group'] == group]
    print(f"\n**** SOFA Score group: {group} ****")

    # 2.11 准备自变量和因变量，去除'sofa_score' 和 'sofa_score_group'
    X = subgroup_data.drop(['ards', 'sofa_score', 'sofa_score_group'], axis=1)
    y = subgroup_data['ards']

    # 2.12 添加常数项（截距）
    X = sm.add_constant(X)

    # 3. 拟合逻辑回归模型
    def fit_logistic_model(X, y):
        try:
            model = sm.Logit(y, X)
            result = model.fit(disp=0)
            return result
        except Exception as e:
            print(f"模型拟合错误：{e}")
            return None

    result = fit_logistic_model(X, y)

    # 4. 显示结果
    if result is not None:
        print(result.summary())

        # 提取 bpri_log 的 OR 和 p 值
        if 'bpri_log' in result.params:
            coef = result.params['bpri_log']
            pval = result.pvalues['bpri_log']
            or_val = np.exp(coef)
            print(f"bpri_log: OR = {or_val:.3f}, p值 = {pval:.3f}")

    else:
        print(f"亚组 {group} 的模型未成功拟合.")

    # 5. 保存结果到CSV文件
    if result is not None:
        # 处理非法字符，替换为下划线以适应文件命名
        safe_group = group.replace("0-6", "0_6").replace("7+", "7_plus")
        output_file = os.path.join(output_dir, f"sofa_score_group_{safe_group}_model_results.csv")
        result_df = result.summary2().tables[1]
        result_df.to_csv(output_file)

# 6. 新增：构造统一模型并添加交互项进行两 SOFA 评分亚组间的 Wald 检验
#    构造 SOFA 分组指示变量：1 表示 '7+' 组，0 表示 '0-6' 组
data_encoded['sofa_group_7plus'] = (data_encoded['sofa_score_group'] == '7+').astype(int)

# 构造 bpri_log 与 SOFA 分组的交互项
data_encoded['bpri_log_sofa'] = data_encoded['bpri_log'] * data_encoded['sofa_group_7plus']

# 构造统一模型的自变量列表，此处选择检验 bpri_log 及其交互项对 ARDS 的影响
X_interact = data_encoded[['bpri_log', 'sofa_group_7plus', 'bpri_log_sofa']]
X_interact = sm.add_constant(X_interact)
y_full = data_encoded['ards']

model_interact = sm.Logit(y_full, X_interact)
result_interact = model_interact.fit(disp=0)

print("\n**** 统一模型（包含交互项）的多因素分析结果 ****")
print(result_interact.summary())

# 对交互项进行 Wald 检验：原假设 bpri_log_sofa 的系数为 0
wald_test_interaction = result_interact.wald_test('bpri_log_sofa = 0')
print("\nWald 检验结果：检验 bpri_log 与 SOFA 分组交互项 (bpri_log_sofa) 是否显著")
print(wald_test_interaction)
