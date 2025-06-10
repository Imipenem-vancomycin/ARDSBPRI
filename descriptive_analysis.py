import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, fisher_exact, ttest_ind, mannwhitneyu, kstest, norm, levene

# 读取数据
file_path = r"D:\pythonproject\saaki\ARDSbpri.csv"
data = pd.read_csv(file_path)

# 检查并处理无效值
print("缺失值统计：")
print(data.isnull().sum())

# 检查数值列中的 inf 值
print("Inf 值统计：")
print(data.select_dtypes(include=[np.number]).apply(lambda x: np.isinf(x).sum()))

# 将 inf 替换为 NaN
data = data.replace([np.inf, -np.inf], np.nan)

# 删除包含 NaN 的行
data = data.dropna()

# 确保 'bpri' 列存在并生成 'bpri_log'
if 'bpri' in data.columns:
    data['bpri_log'] = np.log(data['bpri'] + 1)  # 避免 log(0) 错误
else:
    raise ValueError("数据中缺少 'bpri' 列，无法生成 'bpri_log'")

# 分组
ards_group = data[data['ards'] == 1]
non_ards_group = data[data['ards'] == 0]

# 定义分类变量和连续变量
categorical_vars = [
    "metastatic_solid_tumor", "morality28d", "hospital_expire_flag", "chronic_pulmonary_disease",
    "paraplegia", "cerebrovascular_disease", "hypertension", "arrhythmia", "malignant_cancer",
    "peptic_ulcer_disease", "myocardial_infarct", "gender", "race", "aids", "rheumatic_disease"
]

excluded_vars = [
    'hadm_id', 'stay_id', 'subject_id', 'sofa_time', 'icu_intime', 'icu_outtime',
    'admittime', 'dischtime', 'vis_starttime', 'vis_endtime'
]

continuous_vars = [
    col for col in data.columns
    if col not in categorical_vars + ['ards'] + excluded_vars
]

# 初始化结果存储
descriptive_results = []

# 描述性分析 - 所有患者
for var in categorical_vars + continuous_vars:
    try:
        if var in categorical_vars:
            # 分类变量：计算计数和百分比
            counts = data[var].value_counts()
            percentages = counts / counts.sum() * 100
            all_patients_desc = f"{counts.values[0]} ({percentages.values[0]:.2f}%)"
        else:
            # 连续变量：计算均值 ± 标准差或中位数（IQR）
            if len(data[var].dropna()) > 1:  # 确保样本量大于 1
                if kstest(data[var].dropna(), 'norm', args=(data[var].mean(), data[var].std())).pvalue > 0.05:
                    mean = data[var].mean()
                    std = data[var].std()
                    all_patients_desc = f"{mean:.2f} ± {std:.2f}"
                else:
                    median = data[var].median()
                    iqr = data[var].quantile([0.25, 0.75])
                    all_patients_desc = f"{median:.2f} ({iqr[0.25]:.2f}, {iqr[0.75]:.2f})"
            else:
                all_patients_desc = "NaN"
        
        descriptive_results.append({
            "Variable": var,
            "Type": "All Patients",
            "All Patients": all_patients_desc,
            "ARDS Group": "",
            "Non-ARDS Group": "",
            "P-value": "",
            "Test Method": ""
        })
    except Exception as e:
        print(f"Error processing variable {var} for all patients: {e}")

# 描述性分析 - 分类变量
for var in categorical_vars:
    try:
        # 构建列联表
        contingency_table = pd.crosstab(data[var], data['ards'])
        
        # 检查期望频数，选择卡方检验或 Fisher 精确检验
        if np.any(contingency_table < 5):
            _, p_value = fisher_exact(contingency_table)
        else:
            _, p_value, _, _ = chi2_contingency(contingency_table)
        
        # 计算各组计数和百分比
        ards_count = contingency_table[1]
        non_ards_count = contingency_table[0]
        if len(ards_count) > 0 and len(non_ards_count) > 0:
            ards_percent = ards_count / ards_count.sum() * 100
            non_ards_percent = non_ards_count / non_ards_count.sum() * 100
        else:
            ards_percent, non_ards_percent = np.nan, np.nan
        
        descriptive_results.append({
            "Variable": var,
            "Type": "Categorical",
            "All Patients": "",
            "ARDS Group (n)": ards_count.values,
            "ARDS Group (%)": ards_percent.values,
            "Non-ARDS Group (n)": non_ards_count.values,
            "Non-ARDS Group (%)": non_ards_percent.values,
            "P-value": p_value
        })
    except Exception as e:
        print(f"Error processing categorical variable {var}: {e}")

# 描述性分析 - 连续变量
for var in continuous_vars:
    try:
        # 使用 KS 检验检查正态性
        if len(ards_group[var].dropna()) > 1 and len(non_ards_group[var].dropna()) > 1:
            ks_stat_ards, p_ks_ards = kstest(ards_group[var].dropna(), 'norm', args=(ards_group[var].mean(), ards_group[var].std()))
            ks_stat_non_ards, p_ks_non_ards = kstest(non_ards_group[var].dropna(), 'norm', args=(non_ards_group[var].mean(), non_ards_group[var].std()))
            
            # 检查方差齐性
            _, p_levene = levene(ards_group[var].dropna(), non_ards_group[var].dropna())
            
            # 根据正态性和方差齐性选择检验方法
            if p_ks_ards > 0.05 and p_ks_non_ards > 0.05 and p_levene > 0.05:
                # 正态分布且方差齐性，使用 t 检验
                _, p_value = ttest_ind(ards_group[var].dropna(), non_ards_group[var].dropna())
                test_method = "t-test"
            else:
                # 非正态分布或方差不齐，使用 Mann-Whitney U 检验
                _, p_value = mannwhitneyu(ards_group[var].dropna(), non_ards_group[var].dropna())
                test_method = "Mann-Whitney U"
            
            # 计算均值和标准差或中位数和四分位数
            if test_method == "t-test":
                ards_mean = ards_group[var].mean()
                ards_std = ards_group[var].std()
                non_ards_mean = non_ards_group[var].mean()
                non_ards_std = non_ards_group[var].std()
                ards_desc = f"{ards_mean:.2f} ± {ards_std:.2f}"
                non_ards_desc = f"{non_ards_mean:.2f} ± {non_ards_std:.2f}"
            else:
                ards_median = ards_group[var].median()
                ards_iqr = ards_group[var].quantile([0.25, 0.75])
                non_ards_median = non_ards_group[var].median()
                non_ards_iqr = non_ards_group[var].quantile([0.25, 0.75])
                ards_desc = f"{ards_median:.2f} ({ards_iqr[0.25]:.2f}, {ards_iqr[0.75]:.2f})"
                non_ards_desc = f"{non_ards_median:.2f} ({non_ards_iqr[0.25]:.2f}, {non_ards_iqr[0.75]:.2f})"
        else:
            ards_desc, non_ards_desc, p_value, test_method = "NaN", "NaN", np.nan, "NaN"
        
        descriptive_results.append({
            "Variable": var,
            "Type": "Continuous",
            "All Patients": "",
            "ARDS Group": ards_desc,
            "Non-ARDS Group": non_ards_desc,
            "P-value": p_value,
            "Test Method": test_method
        })
    except Exception as e:
        print(f"Error processing continuous variable {var}: {e}")

# 转为 DataFrame
descriptive_results_df = pd.DataFrame(descriptive_results)

# 确保 P-value 列为数值类型
descriptive_results_df['P-value'] = pd.to_numeric(descriptive_results_df['P-value'], errors='coerce')
descriptive_results_df['P-value'] = descriptive_results_df['P-value'].replace([np.nan, np.inf, -np.inf], 1.0)

# 根据 P 值进行排序
descriptive_results_df = descriptive_results_df.sort_values(by="P-value", ascending=True)

# 打印汇总结果
print("\n\n==== 描述性分析结果汇总 ====")
print(descriptive_results_df)

# 保存结果
output_path = r"D:\pythonproject\ardsbpri\descriptive_analysis_results.csv"
descriptive_results_df.to_csv(output_path, index=False)
print(f"\n描述性分析结果已保存至：{output_path}")
