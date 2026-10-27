# ============================================================
# OASIS Dementia Prediction — EDA + Preprocessing
# Run this in Google Colab after uploading both CSVs
# ============================================================

from google.colab import files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Upload files (run once)
# uploaded = files.upload()   # select oasis_cross-sectional.csv and oasis_longitudinal.csv

# ------------------------------------------------------------
# Step 2: Load both datasets
# ------------------------------------------------------------
df_cross = pd.read_csv("oasis_cross-sectional.csv")   # OASIS-1
df_long  = pd.read_csv("oasis_longitudinal.csv")      # OASIS-2

print("OASIS-1 (cross-sectional) shape:", df_cross.shape)
print("OASIS-2 (longitudinal) shape:", df_long.shape)

print("\nOASIS-1 columns:", df_cross.columns.tolist())
print("OASIS-2 columns:", df_long.columns.tolist())

# ------------------------------------------------------------
# Step 3: Basic inspection
# ------------------------------------------------------------
print("\n--- OASIS-2 info ---")
df_long.info()

print("\n--- OASIS-2 summary stats ---")
print(df_long.describe())

print("\n--- Missing values (OASIS-2) ---")
print(df_long.isnull().sum())

print("\n--- Missing values (OASIS-1) ---")
print(df_cross.isnull().sum())

# ------------------------------------------------------------
# Step 4: Target variable — Group / CDR
# ------------------------------------------------------------
# OASIS-2 has a 'Group' column: Demented, Nondemented, Converted
print("\nGroup value counts (OASIS-2):")
print(df_long['Group'].value_counts())

# We'll simplify 'Converted' into 'Demented' for binary classification,
# since converted patients developed dementia during the study
df_long['Group_Binary'] = df_long['Group'].replace({'Converted': 'Demented'})
print("\nBinary group counts:")
print(df_long['Group_Binary'].value_counts())

# ------------------------------------------------------------
# Step 5: Handle missing values
# ------------------------------------------------------------
# SES and MMSE typically have missing values in OASIS-2
# Justify: fill SES with median (ordinal scale 1-5), MMSE with median too
print("\nMissing before cleaning:")
print(df_long[['SES', 'MMSE']].isnull().sum())

df_long['SES'] = df_long['SES'].fillna(df_long['SES'].median())
df_long['MMSE'] = df_long['MMSE'].fillna(df_long['MMSE'].median())

print("\nMissing after cleaning:")
print(df_long[['SES', 'MMSE']].isnull().sum())

# ------------------------------------------------------------
# Step 6: Encode categorical variables
# ------------------------------------------------------------
df_long['M/F'] = df_long['M/F'].map({'M': 1, 'F': 0})
df_long['Group_Binary_Encoded'] = df_long['Group_Binary'].map({'Nondemented': 0, 'Demented': 1})

# ------------------------------------------------------------
# Step 7: Exploratory visualizations
# ------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

sns.countplot(data=df_long, x='Group_Binary', ax=axes[0,0])
axes[0,0].set_title('Dementia Status Distribution')

sns.histplot(data=df_long, x='Age', hue='Group_Binary', kde=True, ax=axes[0,1])
axes[0,1].set_title('Age Distribution by Group')

sns.boxplot(data=df_long, x='Group_Binary', y='MMSE', ax=axes[0,2])
axes[0,2].set_title('MMSE Score by Group')

sns.boxplot(data=df_long, x='Group_Binary', y='nWBV', ax=axes[1,0])
axes[1,0].set_title('Normalized Brain Volume by Group')

sns.boxplot(data=df_long, x='Group_Binary', y='eTIV', ax=axes[1,1])
axes[1,1].set_title('eTIV by Group')

corr_cols = ['Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF', 'Group_Binary_Encoded']
sns.heatmap(df_long[corr_cols].corr(), annot=True, cmap='coolwarm', ax=axes[1,2])
axes[1,2].set_title('Correlation Heatmap')

plt.tight_layout()
plt.savefig("oasis_eda_overview.png", dpi=150)
plt.show()

# ------------------------------------------------------------
# Step 8: Class imbalance check
# ------------------------------------------------------------
print("\nClass balance:")
print(df_long['Group_Binary_Encoded'].value_counts(normalize=True))

# ------------------------------------------------------------
# Step 9: Save cleaned dataset for the next stage (modeling)
# ------------------------------------------------------------
df_long.to_csv("oasis_longitudinal_cleaned.csv", index=False)
print("\nSaved cleaned dataset: oasis_longitudinal_cleaned.csv")
print("\nReady for the next milestone: model building (16 Oct deadline)")
