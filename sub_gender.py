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
required_columns = continuous_vars + categorical_vars + ['ards', 'bpri_log', 'gender']
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

# 2.9 打印 data_encoded 的列名，确认是否有 gender 列
print("data_encoded 的列名：")
print(data_encoded.columns)

# 2.10 根据 'gender' 列进行分组：男性和女性
# 直接使用原始 'gender' 变量，进行分组
data_male = data_encoded[data_encoded['gender'] == 1]  # 男性组
data_female = data_encoded[data_encoded['gender'] == 0]  # 女性组

# 2.11 创建交互项：bpri_log * gender
data_encoded['bpri_log_gender'] = data_encoded['bpri_log'] * data_encoded['gender']

# 2.12 准备自变量和因变量（分别为男性和女性组）
X_male = data_male.drop(['ards', 'gender'], axis=1)  # 去除性别列
y_male = data_male['ards']
X_female = data_female.drop(['ards', 'gender'], axis=1)  # 去除性别列
y_female = data_female['ards']

# 2.13 添加常数项（截距）
X_male = sm.add_constant(X_male)
X_female = sm.add_constant(X_female)

# 3. 拟合逻辑回归模型（分别对男性和女性进行分析）
def fit_logistic_model(X, y):
    try:
        model = sm.Logit(y, X)
        result = model.fit(disp=0)
        return result
    except Exception as e:
        print(f"模型拟合错误：{e}")
        return None

result_male = fit_logistic_model(X_male, y_male)
result_female = fit_logistic_model(X_female, y_female)

# 4. 显示结果
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

        # 提取bpri_log_gender交互项的OR和P值
        if 'bpri_log_gender' in result.params:
            coef_gender = result.params['bpri_log_gender']
            pval_gender = result.pvalues['bpri_log_gender']
            or_val_gender = np.exp(coef_gender)
            print(f"bpri_log_gender: OR = {or_val_gender:.3f}, p值 = {pval_gender:.3f}")

            # 进行Wald检验
            wald_test = result.wald_test('bpri_log_gender')
            print(f"Wald检验结果: \n{wald_test.summary()}")
        else:
            print("bpri_log_gender 未包含在模型中.")
    else:
        print(f"{group_name}组的模型未成功拟合.")

# 打印男性和女性组的结果
print_model_results(result_male, "男性")
print_model_results(result_female, "女性")

# 5. 保存结果到CSV文件
def save_model_results(result, group_name):
    if result is not None:
        result_df = result.summary2().tables[1]
        result_df.to_csv(os.path.join(output_dir, f"{group_name}_model_results.csv"))

# 保存每组的结果
save_model_results(result_male, "男性")
save_model_results(result_female, "女性")
# 假设你的数据已经加载到 data_encoded
# 生成交互项 bpri_log * gender
data_encoded['bpri_log_gender'] = data_encoded['bpri_log'] * data_encoded['gender']

# 显示数据的列名以确认交互项已被正确创建
print("数据列名：", data_encoded.columns)

# 选择所有自变量，包括新的交互项
X = data_encoded[['sofa_score', 'gcs', 'glucose', 'albumin', 'sapsii', 'avg_wbc', 'avg_lymphocytes',
                  'pao2fio2ratio', 'cirrhosis', 'bilirubin_total', 'sodium', 'bpri_log', 'bpri_log_gender',
                  'calcium', 't', 'map', 'base_excess', 'lactate', 'creatinine', 'ph', 'nmlr', 'wbc',
                  'age', 'sii', 'height', 'potassium', 'rr', 'hr', 'hb', 'bmi', 'charlson_comorbidity_index',
                  'avg_neutrophils_abs', 'metastatic_solid_tumor_1.0', 'chronic_pulmonary_disease_1.0',
                  'paraplegia_1.0', 'cerebrovascular_disease_1.0', 'hypertension_1.0', 'arrhythmia_1.0',
                  'malignant_cancer_1.0', 'race_2.0', 'race_3.0', 'race_4.0', 'race_5.0', 'peptic_ulcer_disease_1.0',
                  'aids_1.0', 'myocardial_infarct_1.0']]

# 增加常数项
X = sm.add_constant(X)

# 目标变量
y = data_encoded['ards']

# 拟合逻辑回归模型
model = sm.Logit(y, X)
result = model.fit()

# 输出回归结果
print(result.summary())

# 提取并显示 bpri_log 和 bpri_log_gender 的相关信息
print(f"bpri_log 和 bpri_log_gender 交互项的回归系数：")
print(f"bpri_log 的系数： {result.params['bpri_log']}, p值： {result.pvalues['bpri_log']}")
print(f"bpri_log_gender 的系数： {result.params['bpri_log_gender']}, p值： {result.pvalues['bpri_log_gender']}")