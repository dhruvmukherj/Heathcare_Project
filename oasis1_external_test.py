# OASIS-1 as a second test set
# Train on OASIS-2 (cleaned file), test on OASIS-1 people who have a CDR score.
# Needs in the Colab file panel: oasis_longitudinal_cleaned.csv, oasis_cross-sectional.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay)

train_df = pd.read_csv("oasis_longitudinal_cleaned.csv")
o1 = pd.read_csv("oasis_cross-sectional.csv")

# In OASIS-1, Educ is a code from 1 to 5. In OASIS-2, EDUC is years of school.
# They are not the same thing, so Education is left out of this test.
print("OASIS-1 Educ values:", sorted(o1['Educ'].dropna().unique()))

# Keep only people who have a CDR score, and ages 60+ (OASIS-2 is 60 to 98)
o1 = o1[o1['CDR'].notna() & (o1['Age'] >= 60)].copy()
o1['M/F'] = o1['M/F'].map({'M': 1, 'F': 0})
# Label rule here: CDR above 0 = demented
o1['Demented'] = (o1['CDR'] > 0).astype(int)

features = ['eTIV', 'nWBV', 'ASF', 'Age', 'SES', 'MMSE', 'M/F']

# Fill missing SES with the OASIS-2 median, same as in training
print("OASIS-1 SES missing, filled with OASIS-2 median:", o1['SES'].isna().sum())
o1['SES'] = o1['SES'].fillna(train_df['SES'].median())

before = len(o1)
o1 = o1.dropna(subset=features)
print("Rows dropped for other missing values:", before - len(o1))

X_train, y_train = train_df[features], train_df['Group_Binary_Encoded']
X_test, y_test = o1[features], o1['Demented']

print("\nOASIS-1 test set size:", len(o1))
print(y_test.value_counts().rename({0: 'Nondemented', 1: 'Demented'}))

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(random_state=42)),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
}

rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1-Score": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, proba),
    })

results = pd.DataFrame(rows).round(3)
print("\n=== Trained on OASIS-2, tested on OASIS-1 ===")
print(results.to_string(index=False))
results.to_csv("oasis1_test_results.csv", index=False)

rf = models["Random Forest"]
cm = confusion_matrix(y_test, rf.predict(X_test))
ConfusionMatrixDisplay(cm, display_labels=['Nondemented', 'Demented']).plot(cmap='Blues')
plt.title("Random Forest on OASIS-1 (trained on OASIS-2)")
plt.savefig("oasis1_confusion_matrix.png", dpi=150)
plt.show()
