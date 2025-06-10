import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

# 定义保存路径
output_dir = r"D:/pythonproject/ardsbpri"  # 请根据需要修改为你的路径

# 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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

# 2.5 选择需要的列（显著变量 + 'ards' + 'bpri_log' + 'gender'）并去重
required_columns = continuous_vars + categorical_vars + ['ards', 'bpri_log', 'gender']
required_columns = list(dict.fromkeys(required_columns))  # 去除重复列
missing_required = set(required_columns) - set(data.columns)
if missing_required:
    raise ValueError(f"数据中缺少以下必要列：{missing_required}")

data = data[required_columns]

# 2.6 处理缺失值 (如果确认没有缺失值，可省略或改为只检查)
data = data.dropna(subset=['ards'])  # 删除 'ards' 缺失的行
data = data.dropna()  # 删除任何缺失值行
# 如果真的无缺失值，这两步不会删掉任何行

# 2.7 对分类变量做独热编码
for var in categorical_vars:
    data[var] = data[var].astype('category')

data_encoded = pd.get_dummies(data, columns=categorical_vars, drop_first=True)

# 确保 'gender' 是数值型；如是字符串，则需手动 map
if data_encoded['gender'].dtype.name == 'category':
    data_encoded['gender'] = data_encoded['gender'].astype(int)
elif data_encoded['gender'].dtype.name == 'object':
    # 如果确实是字符串，比如 'M'/'F'，则自行映射
    # data_encoded['gender'] = data_encoded['gender'].map({'M': 1, 'F': 0})
    pass

# 2.8 将布尔列转为 int
bool_cols = data_encoded.select_dtypes(include=['bool']).columns
data_encoded[bool_cols] = data_encoded[bool_cols].astype(int)

# 2.9 按照 PaO2FiO2 ratio 进行分组
data_encoded['pao2fio2_group'] = pd.cut(
    data['pao2fio2ratio'],
    bins=[0, 100, 200, 300, np.inf],
    labels=['<100', '100-200', '200-300', '>300']
)

# 2.10 剔除 'pao2fio2ratio' 从 data_encoded（因分组用途）
if 'pao2fio2ratio' in data_encoded.columns:
    data_encoded.drop(columns=['pao2fio2ratio'], inplace=True)

# ------------------------------
# 3. 分组回归分析，只转换布尔列为整数，不做全局 to_numeric+dropna
# ------------------------------

subgroup_results = {}
groups = ['<100', '100-200', '200-300', '>300']

for group in groups:
    print(f"\n处理组 {group}...")
    subgroup_data = data_encoded[data_encoded['pao2fio2_group'] == group].copy()

    # 如果该子组本身条数很少，可以提前跳过
    n_sub = len(subgroup_data)
    print(f"组 {group} 原始数据行数：{n_sub}")
    if n_sub < 2:
        print(f"组 {group} 数据量不足，跳过。")
        continue

    # 再次将布尔列变成 int（以防万一）
    bool_cols_sub = subgroup_data.select_dtypes(include=['bool']).columns
    subgroup_data[bool_cols_sub] = subgroup_data[bool_cols_sub].astype(int)

    # 检查是否有 object 列（如果有，需要手动处理或删除）
    object_cols = subgroup_data.select_dtypes(include=['object']).columns
    if len(object_cols) > 0:
        print(f"组 {group} 存在 object 列：{object_cols}. 需要转换或删除，否则模型会出错。")
        # 如确认不需要这些列，可删掉：
        # subgroup_data.drop(columns=object_cols, inplace=True)
        # 或对它们做映射/编码
        # 这里先仅打印提示，如果不处理，会在拟合时报错
        pass

    # 准备 X, y
    y = subgroup_data['ards']
    X = subgroup_data.drop(['ards', 'pao2fio2_group'], axis=1, errors='ignore')

    # 若依旧没有可用列，也无法回归
    if X.shape[1] < 1:
        print(f"组 {group} 无可用自变量，跳过。")
        continue

    # 添加常数项
    X = sm.add_constant(X)

    try:
        model = sm.Logit(y, X)
        result = model.fit()
        subgroup_results[group] = result
        print(f"**{group}组的回归结果:**")
        print(result.summary())

        # 提取 bpri_log 的OR和P值
        if 'bpri_log' in result.params:
            coef = result.params['bpri_log']
            pval = result.pvalues['bpri_log']
            or_val = np.exp(coef)
            print(f"bpri_log: OR = {or_val:.3f}, p值 = {pval:.3f}")
        else:
            print("bpri_log 未包含在模型中.")

    except Exception as e:
        print(f"组 {group} 回归错误: {e}")
for group, result in subgroup_results.items():
    if result is not None:
        result_df = result.summary2().tables[1]

        # 对 group 名做安全处理，去掉特殊字符
        # 例如：'<100' -> 'lt_100', '>300' -> 'gt_300'
        safe_group = group.replace('<', 'lt_').replace('>', 'gt_').replace(':', '_')
        # 还可以继续 replace 其他可能不合法的字符

        csv_filename = f"{safe_group}_model_results.csv"
        result_df.to_csv(os.path.join(output_dir, csv_filename))

print("\n分组模型拟合与结果保存完成。")

# ------------------------------
# 4. 全局交互项分析
# ------------------------------
# 与之前思路相同，用 get_dummies 将 pao2fio2_group 编码成若干 0/1 列
data_encoded_interaction = data_encoded.copy()

data_encoded_interaction = pd.get_dummies(
    data_encoded_interaction,
    columns=['pao2fio2_group'],
    drop_first=True  # 把 <100 当作基线
)

# 再次把布尔列转为 int (防止有剩余)
bool_cols = data_encoded_interaction.select_dtypes(include=['bool']).columns
data_encoded_interaction[bool_cols] = data_encoded_interaction[bool_cols].astype(int)

# 创建交互项 bpri_log * pao2fio2_group_xxx
for col in data_encoded_interaction.columns:
    if col.startswith('pao2fio2_group_'):
        new_col = 'bpri_log_' + col
        data_encoded_interaction[new_col] = (
            data_encoded_interaction['bpri_log'] * data_encoded_interaction[col]
        )

# 准备 X, y
if 'ards' not in data_encoded_interaction.columns:
    raise ValueError("缺少 'ards' 列，无法进行交互项分析。")

X_interaction = data_encoded_interaction.drop(['ards'], axis=1)
y_interaction = data_encoded_interaction['ards']

# 确保没有 object 列
object_cols = X_interaction.select_dtypes(include=['object']).columns
if len(object_cols) > 0:
    print(f"交互项模型中还存在 object 列: {object_cols}, 需转换或删除。")

# 添加常数项
X_interaction = sm.add_constant(X_interaction)

try:
    model_interaction = sm.Logit(y_interaction, X_interaction)
    result_interaction = model_interaction.fit()
    print("\n**包含交互项的模型回归结果:**")
    print(result_interaction.summary())

    # bpri_log 基础项
    if 'bpri_log' in result_interaction.params:
        coef_bpri = result_interaction.params['bpri_log']
        pval_bpri = result_interaction.pvalues['bpri_log']
        print(f"\n基础项 bpri_log 系数 = {coef_bpri:.4f}, p值 = {pval_bpri:.4g}")

    # 交互项
    int_cols = [c for c in result_interaction.params.index if c.startswith('bpri_log_pao2fio2_group_')]
    for ic in int_cols:
        coef_ic = result_interaction.params[ic]
        pval_ic = result_interaction.pvalues[ic]
        print(f"交互项 {ic}: 系数 = {coef_ic:.4f}, p值 = {pval_ic:.4g}")

    # 保存交互项结果
    out_int = result_interaction.summary2().tables[1]
    out_int.to_csv(os.path.join(output_dir, "pao2fio2_interaction_model_results.csv"))

except Exception as e:
    print(f"交互项模型拟合错误: {e}")

print("\n所有分析完成。")
