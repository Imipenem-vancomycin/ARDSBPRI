import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu

# 让 pandas 在打印 DataFrame 时不省略列和内容
pd.set_option('display.max_rows', None)        # 显示所有行
pd.set_option('display.max_columns', None)     # 显示所有列
pd.set_option('display.width', None)           # 自动调节列宽
pd.set_option('display.max_colwidth', None)    # 不限制单列最大宽度
pd.set_option('display.precision', 6)          # 设置小数精度（可根据需要调整）

# 读取数据
file_path = r"D:\pythonproject\saaki\ARDSbpri.csv"
data = pd.read_csv(file_path)

# 确保 'bpri' 列存在并生成 'bpri_log'
if 'bpri' in data.columns:
    data['bpri_log'] = np.log(data['bpri'] + 1)  # 避免 log(0) 错误
else:
    raise ValueError("数据中缺少 'bpri' 列，无法生成 'bpri_log'")

# 设置因变量
y = data['ards']  # 0 是未发生 ARDS，1 是发生 ARDS

# 明确指定分类变量列表（根据您提供的变量）
categorical_vars = [
    "metastatic_solid_tumor", "morality28d", "hospital_expire_flag", "chronic_pulmonary_disease",
    "paraplegia", "cerebrovascular_disease", "hypertension", "arrhythmia", "malignant_cancer",
    "peptic_ulcer_disease", "myocardial_infarct", "gender", "race", "aids", "rheumatic_disease"
]

# 定义连续变量：除分类变量和因变量 'ards' 以及排除变量外的所有变量
excluded_vars = [
    'hadm_id', 'stay_id', 'subject_id', 'sofa_time', 'icu_intime', 'icu_outtime',
    'admittime', 'dischtime', 'vis_starttime', 'vis_endtime'
]
continuous_vars = [
    col for col in data.columns
    if col not in categorical_vars + ['ards'] + excluded_vars
]

print(f"指定的分类变量：{categorical_vars}")
print(f"检测到的连续变量：{continuous_vars}")

# 初始化结果存储
results = []

# 单因素分析 - 连续变量
for var in continuous_vars:
    try:
        # 检查变量的唯一值数量
        unique_vals = data[var].dropna().unique()
        if len(unique_vals) < 2:
            print(f"变量 {var} 缺乏变化（唯一值数量={len(unique_vals)}），跳过分析。")
            continue

        # 分组数据
        group1 = data.loc[y == 0, var]
        group2 = data.loc[y == 1, var]

        # 检查缺失值
        if group1.isnull().all() or group2.isnull().all():
            print(f"变量 {var} 在某一组中全为缺失值，跳过分析。")
            continue

        # 两组均值差异检验
        _, p_value_ttest = ttest_ind(group1, group2, nan_policy='omit')
        _, p_value_mann = mannwhitneyu(group1, group2, alternative='two-sided')

        # 单因素逻辑回归
        X = sm.add_constant(data[var].astype(float))  # 确保为浮点数并添加常数项
        model = sm.Logit(y, X)  # 构建逻辑回归模型
        result = model.fit(disp=0)
        or_val = np.exp(result.params[1])  # OR值
        ci_lower, ci_upper = np.exp(result.conf_int().iloc[1])  # 置信区间
        p_val = result.pvalues[1]  # P值

        results.append({
            "Variable": var,
            "Type": "Continuous",
            "OR": or_val,
            "CI Lower (2.5%)": ci_lower,
            "CI Upper (97.5%)": ci_upper,
            "P-value (Logit)": p_val,
            "P-value (T-test)": p_value_ttest,
            "P-value (Mann-Whitney)": p_value_mann,
            "P-value (Chi2)": np.nan  # 不适用于连续变量
        })

        # 立即打印当前变量的结果
        print(f"【连续变量 {var} 的单因素分析结果】")
        print(results[-1])
        print("-" * 60)

    except Exception as e:
        print(f"Error processing variable {var}: {e}")

# 单因素分析 - 分类变量
for var in categorical_vars:
    try:
        # 确保分类变量是数值型或分类型
        if data[var].dtype not in ['int64', 'float64', 'object', 'category']:
            data[var] = data[var].astype('category')

        unique_values = data[var].dropna().unique()
        num_categories = len(unique_values)

        if num_categories > 2:
            print(f"变量 {var} 具有多于两个类别，正在进行哑变量编码。")
            # 进行哑变量编码，排除第一个类别以避免虚拟变量陷阱
            dummies = pd.get_dummies(data[var], drop_first=True, prefix=var).astype(int)
            for dummy_var in dummies.columns:
                # 确保虚拟变量为整数类型
                X = sm.add_constant(dummies[dummy_var].astype(int))  # 添加常数项
                model = sm.Logit(y, X)
                result = model.fit(disp=0)
                or_val = np.exp(result.params[1])  # OR值
                ci_lower, ci_upper = np.exp(result.conf_int().iloc[1])  # 置信区间
                p_val = result.pvalues[1]  # P值

                # 计算卡方检验的 P 值
                contingency_table = pd.crosstab(dummies[dummy_var], y)
                chi2, p_value_chi2, _, _ = chi2_contingency(contingency_table)

                results.append({
                    "Variable": dummy_var,
                    "Type": "Categorical",
                    "OR": or_val,
                    "CI Lower (2.5%)": ci_lower,
                    "CI Upper (97.5%)": ci_upper,
                    "P-value (Logit)": p_val,
                    "P-value (Chi2)": p_value_chi2,
                    "P-value (T-test)": np.nan,   # 不适用于分类变量
                    "P-value (Mann-Whitney)": np.nan  # 不适用于分类变量
                })

                # 立即打印当前 dummy 变量的结果
                print(f"【分类变量 {var} ({dummy_var}) 的单因素分析结果】")
                print(results[-1])
                print("-" * 60)

        elif num_categories == 2:
            # 二分类变量
            contingency_table = pd.crosstab(data[var], y)
            chi2, p_value_chi2, _, _ = chi2_contingency(contingency_table)

            # 将分类变量转换为二进制（假设第一个类别为0，第二个为1）
            binary_var = var
            if data[var].dtype == 'object' or data[var].dtype.name == 'category':
                data[binary_var] = pd.Categorical(data[var]).codes

            # 检查是否有缺失值
            if data[binary_var].isnull().all():
                print(f"变量 {var} 全部为缺失值，跳过分析。")
                continue

            X = sm.add_constant(data[binary_var].astype(float))  # 添加常数项
            model = sm.Logit(y, X)
            result = model.fit(disp=0)
            or_val = np.exp(result.params[1])  # OR值
            ci_lower, ci_upper = np.exp(result.conf_int().iloc[1])  # 置信区间
            p_val = result.pvalues[1]  # P值

            results.append({
                "Variable": var,
                "Type": "Categorical",
                "OR": or_val,
                "CI Lower (2.5%)": ci_lower,
                "CI Upper (97.5%)": ci_upper,
                "P-value (Logit)": p_val,
                "P-value (Chi2)": p_value_chi2,
                "P-value (T-test)": np.nan,  # 不适用于分类变量
                "P-value (Mann-Whitney)": np.nan  # 不适用于分类变量
            })

            # 立即打印当前二分类变量的结果
            print(f"【分类变量 {var} 的单因素分析结果】")
            print(results[-1])
            print("-" * 60)

        else:
            print(f"变量 {var} 的类别数不符合预期：{num_categories} 类。")

    except Exception as e:
        print(f"Error processing variable {var}: {e}")

# 转为 DataFrame 并保存结果
results_df = pd.DataFrame(results)

# 确保所有必要的 P 值列都存在
expected_columns = [
    "P-value (Logit)",
    "P-value (T-test)",
    "P-value (Mann-Whitney)",
    "P-value (Chi2)"
]
for col in expected_columns:
    if col not in results_df.columns:
        results_df[col] = np.nan

# 根据 "P-value (Logit)" 进行排序
results_df = results_df.sort_values(by="P-value (Logit)", ascending=True)

# 最后打印汇总结果
print("\n\n==== 单因素分析结果汇总（按 Logit P 值排序） ====")
print(results_df)

# 保存结果
output_path = r"D:\pythonproject\ardsbpri\single_factor_full_results_ards_bpri.csv"
results_df.to_csv(output_path, index=False)
print(f"\n单因素分析结果已保存至：{output_path}")
