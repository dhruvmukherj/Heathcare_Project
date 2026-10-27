# OASIS model comparison: Logistic Regression, Random Forest, SVM
# Same 7 features as the final model (no education).
# Visits of one patient always stay together (grouped by Subject ID), and the
# whole thing is repeated with 5 different seeds so one lucky split cannot
# decide the result.
# Needs oasis_longitudinal_cleaned.csv in the Colab file panel.
# Saves: model_comparison_results.csv, roc_curve_comparison.png, confusion_matrix.png

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, ConfusionMatrixDisplay)

df = pd.read_csv("oasis_longitudinal_cleaned.csv")
features = ['eTIV', 'nWBV', 'ASF', 'Age', 'SES', 'MMSE', 'M/F']
X, y, groups = df[features], df['Group_Binary_Encoded'], df['Subject ID']

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(random_state=42)),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
}

all_rows = []
oof_seed0 = {}          # out-of-fold probabilities from seed 0, used for the plots

for seed in range(5):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for name, model in models.items():
        proba = cross_val_predict(model, X, y, cv=cv, groups=groups,
                                  method='predict_proba')[:, 1]
        pred = (proba >= 0.5).astype(int)
        all_rows.append({
            "Model": name, "seed": seed,
            "Accuracy": accuracy_score(y, pred),
            "Precision": precision_score(y, pred),
            "Recall": recall_score(y, pred),
            "F1-Score": f1_score(y, pred),
            "ROC-AUC": roc_auc_score(y, proba),
        })
        if seed == 0:
            oof_seed0[name] = proba

runs = pd.DataFrame(all_rows)
summary = runs.groupby("Model")[["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]].mean()
summary["AUC lowest seed"] = runs.groupby("Model")["ROC-AUC"].min()
summary["AUC highest seed"] = runs.groupby("Model")["ROC-AUC"].max()
summary = summary.round(3).reset_index()

print("=== Every visit of a patient kept together, 5-fold CV, average of 5 seeds ===")
print(summary.to_string(index=False))
summary.to_csv("model_comparison_results.csv", index=False)

# ROC curves (seed 0, out-of-fold predictions)
plt.figure(figsize=(7, 6))
for name, proba in oof_seed0.items():
    fpr, tpr, _ = roc_curve(y, proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y, proba):.3f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison (cross-validated)")
plt.legend()
plt.savefig("roc_curve_comparison.png", dpi=150)
plt.show()

# Confusion matrix for the model with the best average AUC (seed 0 predictions)
best = summary.loc[summary['ROC-AUC'].idxmax(), 'Model']
cm = confusion_matrix(y, (oof_seed0[best] >= 0.5).astype(int))
ConfusionMatrixDisplay(cm, display_labels=['Nondemented', 'Demented']).plot(cmap='Blues')
plt.title(f"Confusion Matrix: {best} (cross-validated)")
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("Best model by average ROC-AUC:", best)
