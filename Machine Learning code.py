import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import joblib
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, auc, brier_score_loss
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import optuna
from ctgan import CTGAN
import os

# ====================== 全局图像格式 ======================
save_dir = 'D:/pythonproject/ardsbpri/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 单栏与双栏宽度（英寸）
single_col_width = 85 / 25.4   # 约3.35英寸
double_col_width = 180 / 25.4  # 约7.09英寸

# 现在将 ROC 图宽度改为 360 mm (≈14.17英寸)，保持16:9比例
roc_width_mm = 360
roc_width_in = roc_width_mm / 25.4          # ~14.17英寸
roc_height_in = roc_width_in * (9.0 / 16.0) # 16:9
figsize_roc = (roc_width_in, roc_height_in)

# 特征重要性图的宽度仍用双栏 + 适当纵横比 (例如1.3)
aspect_ratio_importance = 1.3

# Matplotlib 全局参数
plt.rcParams.update({
    'font.size': 8,          # 基准字号 8pt
    'font.family': 'Arial',  # 无衬线字体
    'axes.titlesize': 10,    # 标题字号
    'axes.labelsize': 9,     # 轴标签字号
    'xtick.labelsize': 8,    # X轴刻度
    'ytick.labelsize': 8,    # Y轴刻度
    'legend.fontsize': 8,    # 图例字号
    'lines.linewidth': 2,    # 线宽≥2pt
    'axes.linewidth': 2.0,   # 坐标轴线宽≥2pt
    'figure.dpi': 600,       # 显示时的 DPI
    'savefig.dpi': 600,      # 保存位图时的 DPI
    'savefig.bbox': 'tight', # 自动裁剪空白
    'pdf.fonttype': 42       # 确保 PDF 中可嵌入字体
})

def save_figures(fig, filename):
    """统一保存：同时输出 PDF（矢量）和 PNG（位图）"""
    fig.savefig(os.path.join(save_dir, f"{filename}.pdf"), format='pdf')
    fig.savefig(os.path.join(save_dir, f"{filename}.png"), dpi=600)
    plt.close(fig)

# ==============================================================

# ========== 数据加载 ==========
file_path = r"D:\\pythonproject\\saaki\\ARDSbpri.csv"
data = pd.read_csv(file_path)

# 确保 'bpri' 列存在并生成 'bpri_log'
if 'bpri' in data.columns:
    data['bpri_log'] = np.log(data['bpri'] + 1)
else:
    raise ValueError("数据中缺少 'bpri' 列，无法生成 'bpri_log'")

# 因变量
y = data['ards']  # 0 = 未发生 ARDS, 1 = 发生 ARDS

# 分类变量（含 race）
categorical_vars = [
    "metastatic_solid_tumor", "chronic_pulmonary_disease",
    "paraplegia", "cerebrovascular_disease", "hypertension", "arrhythmia", "malignant_cancer",
    "peptic_ulcer_disease", "myocardial_infarct", "gender", "aids", "rheumatic_disease", "race"
]

# race 做 one-hot
data = pd.get_dummies(data, columns=["race"], drop_first=True)

# 连续变量：排除分类、目标以及部分无关列
excluded_vars = [
    'hadm_id', 'stay_id', 'subject_id', 'sofa_time', 'icu_intime', 'icu_outtime', 'admittime', 'dischtime',
    "morality28d", "hospital_expire_flag",
    'vis_starttime', 'vis_endtime', 'icu_los', 'platelet', 'nlr', 'dnlr', 'mlr', 'nmlr', 'apsiii_score',
    'avg_neutrophils', 'avg_monocytes', 'avg_monocytes_abs', 'avg_lymphocytes_abs', 'avg_wbc',
    'avg_neutrophils_abs', 'pt', 'weight', 'pco2', 'base_excess', 'hematocrit', 'height', 'rbc',
    'oasis_score', 'early_aki', 'aki_stage', 'sbp', 'dbp', 'ptt', 'wbc', 'bpri'
]
continuous_vars = [col for col in data.columns if col not in categorical_vars + ['ards'] + excluded_vars]

# 标准化
scaler = StandardScaler()
data[continuous_vars] = scaler.fit_transform(data[continuous_vars])

# ========== 打印/保存 VIF ==========
pd.options.display.max_rows = None
X_vif = pd.DataFrame(data, columns=continuous_vars)
vif_data = pd.DataFrame()
vif_data["feature"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
print("完整 VIF 检查结果:")
print(vif_data)
vif_data.to_csv(os.path.join(save_dir, 'vif_results_full.csv'), index=False)

# 检查高 VIF 特征
high_vif_features = vif_data[vif_data["VIF"] > 10]
if not high_vif_features.empty:
    print("以下特征具有较高的 VIF 值（>10）：")
    print(high_vif_features)

# ========== 划分数据集 7:3 ==========
X = data[continuous_vars + [col for col in categorical_vars if col not in ['ards', 'race']]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
y_train = y_train.astype(int)
y_test = y_test.astype(int)

# ========== CTGAN 生成合成数据 ==========
ctgan = CTGAN(epochs=300)
synth_data = pd.concat([X_train, y_train], axis=1)
categorical_features = [col for col in categorical_vars if col in X_train.columns]
ctgan.fit(synth_data, discrete_columns=categorical_features)
synth_sample = ctgan.sample(len(X_train))
X_synth = synth_sample[X_train.columns]
y_synth = synth_sample['ards'].astype(int)

# 合并原始和合成数据
X_train_resampled = pd.concat([X_train, X_synth])
y_train_resampled = pd.concat([y_train, y_synth]).astype(int)

# 10折交叉
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# ========== 定义模型及超参数 ==========
models = {
    'LightGBM': lgb.LGBMClassifier(),
    'RandomForest': RandomForestClassifier(),
    'XGBoost': xgb.XGBClassifier(),
    'CatBoost': cb.CatBoostClassifier(verbose=0),
    'KNN': KNeighborsClassifier()
}

def objective(trial, model_name):
    """Optuna 目标函数"""
    if model_name == 'LightGBM':
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300)
        }
        model = lgb.LGBMClassifier(**params)
    elif model_name == 'RandomForest':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10)
        }
        model = RandomForestClassifier(**params)
    elif model_name == 'XGBoost':
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
        }
        model = xgb.XGBClassifier(**params)
    elif model_name == 'CatBoost':
        params = {
            'iterations': trial.suggest_int('iterations', 50, 300),
            'depth': trial.suggest_int('depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10)
        }
        model = cb.CatBoostClassifier(**params, verbose=0)
    elif model_name == 'KNN':
        params = {
            'n_neighbors': trial.suggest_int('n_neighbors', 3, 20),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski'])
        }
        model = KNeighborsClassifier(**params)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model.fit(X_train_resampled, y_train_resampled)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_pred_prob)

# ========== 训练模型并保存 ==========

best_models = {}
results = {}

# ------------------ 创建更宽的 ROC 图 (宽=360 mm) -------------------
fig_overall, ax_overall = plt.subplots(figsize=figsize_roc, dpi=600)

for model_name in models.keys():
    print(f"正在优化模型: {model_name}")
    try:
        # Optuna 搜索
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, model_name), n_trials=20)

        best_params = study.best_params
        print(f"最佳超参数: {best_params}")

        # 根据最佳超参数重新训练模型
        if model_name == 'LightGBM':
            best_model = lgb.LGBMClassifier(**best_params)
        elif model_name == 'RandomForest':
            best_model = RandomForestClassifier(**best_params)
        elif model_name == 'XGBoost':
            best_model = xgb.XGBClassifier(**best_params)
        elif model_name == 'CatBoost':
            best_model = cb.CatBoostClassifier(**best_params, verbose=0)
        elif model_name == 'KNN':
            best_model = KNeighborsClassifier(**best_params)

        best_model.fit(X_train_resampled, y_train_resampled)

        # 等温校准
        calibrated_model = CalibratedClassifierCV(best_model, method='isotonic', cv='prefit')
        calibrated_model.fit(X_train_resampled, y_train_resampled)

        # 保存最佳模型
        best_models[model_name] = calibrated_model
        joblib.dump(calibrated_model, os.path.join(save_dir, f"{model_name}_best_model.pkl"))
        print(f"{model_name} 模型训练并保存完毕！")

        # 计算评估指标
        y_pred_prob = calibrated_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc_val = auc(fpr, tpr)

        y_pred = calibrated_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        brier = brier_score_loss(y_test, y_pred_prob)

        # 打印评估
        print(f"Model: {model_name}")
        print(f"AUC: {roc_auc_val:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Brier Score: {brier:.4f}")
        print("-" * 50)

        # 结果入字典
        results[model_name] = {
            'AUC': roc_auc_val,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Brier Score': brier
        }

        # 绘制到总体 ROC
        ax_overall.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc_val:.2f})')

        # 单独 ROC: 同样使用宽=360 mm的尺寸
        fig_individual, ax_individual = plt.subplots(figsize=figsize_roc, dpi=600)
        ax_individual.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc_val:.2f}')
        ax_individual.plot([0, 1], [0, 1], color='grey', linestyle='--')
        ax_individual.set_title(f'{model_name} ROC Curve')
        ax_individual.set_xlabel('False Positive Rate')
        ax_individual.set_ylabel('True Positive Rate')
        ax_individual.legend(loc='lower right')
        ax_individual.grid(True)
        save_figures(fig_individual, f"{model_name}_roc_curve")

        # ========== 特征重要性 ==========
        if model_name in ['LightGBM', 'RandomForest', 'XGBoost', 'CatBoost']:
            if model_name == 'CatBoost':
                feature_importances = best_model.get_feature_importance()
            else:
                feature_importances = best_model.feature_importances_

            importance_df = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': feature_importances
            }).sort_values(by='Importance', ascending=False)

            # “旧列名→新列名”映射
            feature_name_map = {
                'hr': 'HR', 'rr': 'RR', 'sbp': 'SBP', 'dbp': 'DBP', 'map': 'MAP',
                'vis': 'VIS', 'bpri_log': 'BPRI Log', 'sapsii': 'SAPSII',
                'sofa_score': 'SOFA Score', 'gcs': 'GCS', 'age': 'Age', 'icu_los': 'ICU Los',
                'wbc': 'WBC', 'hematocrit': 'Hematocrit', 'hb': 'Hb', 'rbc': 'RBC',
                'avg_lymphocytes': 'lymphocytes abs', 'avg_platelet': 'Platelet',
                'avg_monocytes_abs': 'monocytes abs',
                'avg_neutrophils_abs': 'neutrophils abs',
                'lymphocytes_%': 'Lymphocytes%', 'monocytes_%': 'Monocytes%',
                'neutrophils_%': 'Neutrophils%',
                'nlr': 'nlr', 'dnlr': 'dnlr', 'mlr': 'mlr', 'nmlr': 'nmlr',
                'siri': 'siri', 'sii': 'sii',
                'bmi': 'BMI', 'height': 'Height', 'weight': 'Weight',
                'ph': 'PH', 'pco2': 'PCO2', 'po2': 'PO2', 'pao2fio2ratio': 'Pao2Fio2ratio',
                'lactate': 'Lactate', 'base_excess': 'Base Excess',
                'potassium': 'Potassium', 'creatinine': 'Creatinine',
                'charlson_comorbidity_index': 'Charlson Comorbidity Index',
                't': 'T', 'glucose': 'Glucose', 'bilirubin_total': 'Total Bilirubin',
                'albumin': 'Albumin', 'pt': 'PT', 'ptt': 'PTT', 'alt': 'ALT',
                'alp': 'ALP', 'inr': 'INR', 'ast': 'AST', 'calcium': 'calcium',
                'sodium': 'sodium',
                'metastatic_solid_tumor': 'Metastatic Solid Tumor',
                'morality28d': 'Morality28d', 'hospital_expire_flag': 'Hospital Expire Flag',
                'chronic_pulmonary_disease': 'Chronic Pulmonary Disease',
                'paraplegia': 'Paraplegia',
                'cerebrovascular_disease': 'Cerebrovascular Disease',
                'hypertension': 'Hypertension', 'arrhythmia': 'Arrhythmia',
                'malignant_cancer': 'Malignant Cancer',
                'peptic_ulcer_disease': 'Peptic Ulcer Disease',
                'myocardial_infarct': 'Myocardial Infarct',
                'gender': 'Gender', 'aids': 'Aids', 'rheumatic_disease': 'Rheumatic Disease',
                'diabetes': 'Diabetes', 'cirrhosis': 'Cirrhosis',
                'race_1.0': 'Race Black',
                'race_2.0': 'Race White',
                'race_3.0': 'Race Hispanic',
                'race_4.0': 'Race Asian',
                'race_5.0': 'Race Other'
            }
            importance_df['Feature'] = importance_df['Feature'].replace(feature_name_map)

            # 使用双栏宽度 + 等比例加大 (例如1.3) 作为特征重要性图
            fig_imp_width = double_col_width              # ~7.09 in
            fig_imp_height = fig_imp_width * aspect_ratio_importance  # ~7.09 * 1.3 = 9.21 in

            # (A) 前 20 特征图
            top20_df = importance_df.iloc[:20].copy()
            fig_top20, ax_top20 = plt.subplots(
                figsize=(fig_imp_width, fig_imp_height), dpi=300
            )
            ax_top20.barh(top20_df["Feature"], top20_df["Importance"], color='skyblue')
            ax_top20.set_title(f'{model_name} Feature Importance (Top 20)')
            ax_top20.set_xlabel('Feature Importance')
            ax_top20.set_ylabel('Features')
            ax_top20.invert_yaxis()
            plt.subplots_adjust(left=0.35)  # 增加左边距
            save_figures(fig_top20, f"{model_name}_feature_importance_top20")

            # (B) 全部特征图
            fig_all, ax_all = plt.subplots(
                figsize=(fig_imp_width, fig_imp_height), dpi=300
            )
            ax_all.barh(importance_df["Feature"], importance_df["Importance"], color='skyblue')
            ax_all.set_title(f'{model_name} Feature Importance (All Features)')
            ax_all.set_xlabel('Feature Importance')
            ax_all.set_ylabel('Features')
            ax_all.invert_yaxis()
            plt.subplots_adjust(left=0.35)
            save_figures(fig_all, f"{model_name}_feature_importance_all")

    except Exception as e:
        print(f"模型 {model_name} 优化和训练过程中出现错误: {e}")

# ========== 总体 ROC 收尾 ==========
ax_overall.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.50)')
ax_overall.set_xlabel('False Positive Rate')
ax_overall.set_ylabel('True Positive Rate')
ax_overall.set_title('Overall ROC Curve for All Models')
ax_overall.legend(loc='lower right')
ax_overall.grid(True)
save_figures(fig_overall, 'overall_roc_curve')

# ========== 保存评估结果 ==========
with open(os.path.join(save_dir, 'model_evaluation_results.txt'), 'w') as f:
    for model_name, metrics in results.items():
        f.write(f"Model: {model_name}\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
        f.write("\n")

print("全部流程结束！所有图表与模型已保存至：", save_dir)
