import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from patsy import dmatrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, confusion_matrix,
    accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef
)
from statsmodels.tools.sm_exceptions import PerfectSeparationError
import os

width_mm = 360
aspect_ratio = 16 / 9
height_mm = width_mm / aspect_ratio
mm_to_inch = 1 / 25.4
width_inch = width_mm * mm_to_inch
height_inch = height_mm * mm_to_inch
dpi = 600
min_font_size = 12
line_width = 3

plt.rcParams['font.size'] = min_font_size
plt.rcParams['lines.linewidth'] = line_width
sns.set_style("whitegrid")
plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

def find_best_cutpoint(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    J = tpr - fpr
    ix = np.argmax(J)
    best_threshold = thresholds[ix]
    best_J = J[ix]
    return best_threshold, best_J, fpr, tpr, thresholds

def fit_rcs_logistic(df, y_col, x_cols, rcs_col=None, df_spline=4,
                     plot_label="", plot_color="blue"):
    y = df[y_col].values
    other_cols = [c for c in x_cols if c != rcs_col]

    if rcs_col is not None:
        X_spline = dmatrix(f"cr({rcs_col}, df={df_spline}) - 1",
                           {rcs_col: df[rcs_col]},
                           return_type='dataframe')
        X_others = df[other_cols].copy() if len(other_cols) > 0 else pd.DataFrame()
        X_final = pd.concat([X_spline, X_others], axis=1) if not X_others.empty else X_spline
    else:
        X_final = df[x_cols].copy()

    if 'Intercept' in X_final.columns:
        X_final = X_final.drop(columns=['Intercept'])
    X_final = sm.add_constant(X_final, prepend=True, has_constant='add')

    print(f"Design matrix for {plot_label}:")
    print(X_final.head())
    print(X_final.columns.tolist())

    try:
        model = sm.Logit(y, X_final)
        result = model.fit(disp=False, maxiter=100, method='newton')
    except PerfectSeparationError:
        print(f"Perfect separation encountered in {plot_label}. Skipping.")
        return None, None, None
    except Exception as e:
        print(f"Error in {plot_label}: {e}")
        return None, None, None

    print(f"\n=== {plot_label} ===")
    print(result.summary())

    if rcs_col is not None:
        x_min, x_max = df[rcs_col].min(), df[rcs_col].max()
        x_seq = np.linspace(x_min, x_max, 100)

        fixed_vals = {}
        for oc in other_cols:
            if df[oc].dtype.kind in ['i','f']:
                fixed_vals[oc] = df[oc].mean()
            else:
                fixed_vals[oc] = df[oc].mode()[0]

        tmp_spline = dmatrix(f"cr({rcs_col}, df={df_spline}) - 1",
                             {rcs_col: x_seq}, return_type='dataframe')
        tmp_data = tmp_spline.copy()
        for oc in other_cols:
            tmp_data[oc] = fixed_vals[oc]

        if 'Intercept' in tmp_data.columns:
            tmp_data = tmp_data.drop(columns=['Intercept'])
        tmp_data = sm.add_constant(tmp_data, prepend=True, has_constant='add')

        pred_prob = result.predict(tmp_data)
        return result, x_seq, pred_prob
    else:
        return result, None, None

def find_bpri_log_cutoff_for_probability(
    model_result, df, rcs_col='bpri_log', df_spline=4,
    target_prob=0.5, other_cols=None
):
    if other_cols is None:
        other_cols = []

    x_min, x_max = df[rcs_col].min(), df[rcs_col].max()
    x_seq = np.linspace(x_min, x_max, 200)

    fixed_vals = {}
    for oc in other_cols:
        if df[oc].dtype.kind in ['i','f']:
            fixed_vals[oc] = df[oc].mean()
        else:
            fixed_vals[oc] = df[oc].mode()[0]

    new_rows = []
    for v in x_seq:
        tmp = fixed_vals.copy()
        tmp[rcs_col] = v
        new_rows.append(tmp)
    new_data = pd.DataFrame(new_rows)

    X_spline = dmatrix(f"cr({rcs_col}, df={df_spline}) - 1", new_data, return_type='dataframe')
    if len(other_cols) > 0:
        X_others = new_data[other_cols]
        X_final = pd.concat([X_spline, X_others], axis=1)
    else:
        X_final = X_spline.copy()

    if 'Intercept' in X_final.columns:
        X_final = X_final.drop(columns=['Intercept'])
    X_final = sm.add_constant(X_final, prepend=True, has_constant='add')

    pred_prob = model_result.predict(X_final)

    idx_closest = np.argmin(np.abs(pred_prob - target_prob))
    best_bpri_log = x_seq[idx_closest]
    best_pred = pred_prob[idx_closest]
    return best_bpri_log, best_pred

data_path = "D:/pythonproject/saaki/ARDSbpri.csv"
data = pd.read_csv(data_path)

required_cols = [
    'ards','bpri','race','bmi','age','gender','sofa_score','sapsii',
    'gcs','charlson_comorbidity_index','albumin','ph','chronic_pulmonary_disease','lactate'
]
for col in required_cols:
    if col not in data.columns:
        raise ValueError(f"Missing required column: {col}")

data = data.dropna(subset=required_cols)

if 'bpri_log' not in data.columns:
    data['bpri_log'] = np.log1p(data['bpri'])

data['ards'] = data['ards'].astype(int)

categorical_cols = ['race', 'gender', 'chronic_pulmonary_disease']
dummy_cols = [col for col in categorical_cols
              if data[col].dtype == object or data[col].dtype.name == 'category']

if dummy_cols:
    data = pd.get_dummies(data, columns=dummy_cols, drop_first=True)

race_dummies = [col for col in data.columns if col.startswith('race_')]
gender_dummies = [col for col in data.columns if col.startswith('gender_')]
chronic_pulmonary_disease_dummies = [col for col in data.columns if col.startswith('chronic_pulmonary_disease_')]
dummy_variables = race_dummies + gender_dummies + chronic_pulmonary_disease_dummies

multi_vars = ['bmi','age','sofa_score'] + dummy_variables
more_vars = multi_vars + [
    'sapsii','gcs','charlson_comorbidity_index','albumin','ph','lactate'
]

continuous_cols = ['bmi', 'age', 'sofa_score', 'sapsii',
                   'gcs', 'charlson_comorbidity_index',
                   'albumin', 'ph', 'lactate']
scaler = StandardScaler()
data[continuous_cols] = scaler.fit_transform(data[continuous_cols])

model1_bprilog, xseq1_bprilog, pprob1_bprilog = fit_rcs_logistic(
    df=data,
    y_col='ards',
    x_cols=['bpri_log'],
    rcs_col='bpri_log',
    df_spline=4,
    plot_label="Model1-BPRI Log (RCS)",
    plot_color="red"
)

model2_bprilog, xseq2_bprilog, pprob2_bprilog = fit_rcs_logistic(
    df=data,
    y_col='ards',
    x_cols=['bpri_log'] + multi_vars,
    rcs_col='bpri_log',
    df_spline=4,
    plot_label="Model2-BPRI Log (RCS)",
    plot_color="blue"
)

model3_bprilog, xseq3_bprilog, pprob3_bprilog = fit_rcs_logistic(
    df=data,
    y_col='ards',
    x_cols=['bpri_log'] + more_vars,
    rcs_col='bpri_log',
    df_spline=4,
    plot_label="Model3-BPRI Log (RCS)",
    plot_color="green"
)

y_true = data['ards'].values

def do_roc_for_model(model_result, df, other_cols, df_spline=4, rcs_col='bpri_log'):
    from patsy import dmatrix
    X_spline = dmatrix(f"cr({rcs_col}, df={df_spline}) - 1",
                       {rcs_col: df[rcs_col]},
                       return_type='dataframe')
    X_others = df[other_cols].copy() if len(other_cols) > 0 else pd.DataFrame()
    X_final = pd.concat([X_spline, X_others], axis=1) if not X_others.empty else X_spline
    if 'Intercept' in X_final.columns:
        X_final.drop(columns=['Intercept'], inplace=True)
    X_final = sm.add_constant(X_final, prepend=True, has_constant='add')

    y_scores = model_result.predict(X_final)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    J = tpr - fpr
    ix = np.argmax(J)
    best_threshold = thresholds[ix]
    best_J = J[ix]
    return best_threshold, best_J, fpr, tpr, thresholds, roc_auc, y_scores

model_info = [
    ("Model1", model1_bprilog, [],            4, "red"),
    ("Model2", model2_bprilog, multi_vars,    4, "blue"),
    ("Model3", model3_bprilog, more_vars,     4, "green"),
]

cutoff_data = {}
for mname, mres, mcols, dfspline, mcolor in model_info:
    if mres is None:
        print(f"{mname} is None, skip.\n")
        continue

    best_threshold, best_J, fpr, tpr, thr, roc_auc, y_scores = do_roc_for_model(
        model_result=mres,
        df=data,
        other_cols=mcols,
        df_spline=dfspline,
        rcs_col='bpri_log'
    )

    b_log_cut, prob_cut = find_bpri_log_cutoff_for_probability(
        model_result=mres,
        df=data,
        rcs_col='bpri_log',
        df_spline=dfspline,
        target_prob=best_threshold,
        other_cols=mcols
    )

    bpri_cut = np.expm1(b_log_cut)

    cutoff_data[mname] = (b_log_cut, bpri_cut, prob_cut, best_threshold, best_J, roc_auc)

    print(f"\n=== {mname} ROC ===")
    print(f"AUC={roc_auc:.4f}, best_threshold(prob)={best_threshold:.4f}, Youden's J={best_J:.4f}")
    print(f"在均值/众数条件下, BPRI Log_cut={b_log_cut:.3f} -> BPRI_cut={bpri_cut:.3f}, 对应预测概率≈{prob_cut:.3f}")

save_path = r'D:\pythonproject\ardsbpri\RCS'
if not os.path.exists(save_path):
    os.makedirs(save_path)

fig, ax = plt.subplots(figsize=(width_inch, height_inch), dpi=dpi)
if xseq1_bprilog is not None and pprob1_bprilog is not None:
    ax.plot(xseq1_bprilog, pprob1_bprilog, label="Model1-BPRI Log", color="red")
if xseq2_bprilog is not None and pprob2_bprilog is not None:
    ax.plot(xseq2_bprilog, pprob2_bprilog, label="Model2-BPRI Log", color="blue")
if xseq3_bprilog is not None and pprob3_bprilog is not None:
    ax.plot(xseq3_bprilog, pprob3_bprilog, label="Model3-BPRI Log", color="green")

for mname, mcolor in [("Model1","red"),("Model2","blue"),("Model3","green")]:
    if mname in cutoff_data:
        b_log_cut, bpri_cut, prob_cut, best_threshold, best_J, roc_auc = cutoff_data[mname]
        ax.scatter(b_log_cut, prob_cut, s=80, c=mcolor, edgecolor='k', zorder=10,
                    label=f"{mname} cutoff\n(BPRI Log={b_log_cut:.2f}, BPRI={bpri_cut:.2f})")

plt.xlabel("BPRI Log")
plt.ylabel("Predicted Probability of ARDS")
plt.title("Comparison of 3 RCS Curves (BPRI Log) + Best Probability Cutoffs")
plt.legend(loc="best")
plt.savefig(os.path.join(save_path, 'rcs_curves.pdf'), format='pdf', dpi=dpi)
plt.savefig(os.path.join(save_path, 'rcs_curves.png'), format='png', dpi=dpi)
plt.show()

mres = model3_bprilog
if mres is not None and "Model3" in cutoff_data:
    (b_log_cut, bpri_cut, prob_cut, best_threshold, best_J, roc_auc) = cutoff_data["Model3"]
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fig, ax = plt.subplots(figsize=(width_inch, height_inch), dpi=dpi)
    ax.plot(fpr, tpr, color='green', label=f'Model3 ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
    ax.scatter(fpr[np.argmax(tpr - fpr)], tpr[np.argmax(tpr - fpr)],
               color='red', zorder=10, label='Best threshold')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Model3 ROC Curve')
    ax.legend(loc='best')
    plt.savefig(os.path.join(save_path, 'model3_roc.pdf'), format='pdf', dpi=dpi)
    plt.savefig(os.path.join(save_path, 'model3_roc.png'), format='png', dpi=dpi)
    plt.show()

    print(f"\n[Model3-BPRI Log] best_threshold={best_threshold:.4f}, BPRI Log_cut={b_log_cut:.2f}, BPRI_cut={bpri_cut:.2f}, prob={prob_cut:.3f}")

print("\n=== Done. ===")
