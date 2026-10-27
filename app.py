import joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Dementia Risk Predictor", layout="centered")


@st.cache_resource
def load_model():
    bundle = joblib.load("model.pkl")
    explainer = shap.TreeExplainer(bundle["model"])
    return bundle, explainer


bundle, explainer = load_model()
model = bundle["model"]
features = bundle["features"]
imaging = bundle["imaging"]
clinical = bundle["clinical"]
ranges = bundle["ranges"]

NAMES = {"eTIV": "eTIV", "nWBV": "nWBV", "ASF": "ASF", "Age": "Age",
         "SES": "SES", "MMSE": "MMSE", "M/F": "Sex"}

st.title("Dementia Risk Predictor")
st.write(
    "A small course project (Healthcare Analytics, CA-III). Enter a person's "
    "brain scan measurements and basic clinical details, and the model gives "
    "an estimate of whether they belong to the demented group, along with "
    "which inputs pushed the estimate up or down."
)
st.caption("This is for learning only. It is not a medical tool and should "
           "never be used to diagnose anyone.")

# ---------------- inputs ----------------
left, right = st.columns(2)

with left:
    st.subheader("Brain scan measurements")
    etiv = st.slider("eTIV (estimated total intracranial volume)",
                     int(ranges["eTIV"][0]), int(ranges["eTIV"][1]), 1480)
    nwbv = st.slider("nWBV (normalized whole brain volume)",
                     round(ranges["nWBV"][0], 2), round(ranges["nWBV"][1], 2),
                     0.73, step=0.001, format="%.3f")
    asf = bundle["asf_const"] / etiv
    st.caption(f"ASF is worked out from eTIV: {asf:.3f}")

with right:
    st.subheader("Patient details")
    age = st.slider("Age", int(ranges["Age"][0]), int(ranges["Age"][1]), 75)
    sex = st.selectbox("Sex", ["Female", "Male"])
    ses = st.slider("Socioeconomic status (1 = highest, 5 = lowest)", 1, 5, 2)
    mmse = st.slider("MMSE score (out of 30)", 0, 30, 28)

row = pd.DataFrame([{
    "eTIV": etiv, "nWBV": nwbv, "ASF": asf, "Age": age,
    "SES": ses, "MMSE": mmse, "M/F": 1 if sex == "Male" else 0,
}], columns=features)

# ---------------- prediction ----------------
prob = float(model.predict_proba(row)[0, 1])

st.divider()
st.subheader("Result")
c1, c2 = st.columns(2)
c1.metric("Chance of demented group", f"{prob * 100:.0f}%")
c2.metric("Model's call", "Demented" if prob >= 0.5 else "Nondemented")
st.progress(prob)
if 0.4 <= prob <= 0.6:
    st.info("The estimate is close to 50%, so the model is not sure about "
            "this one.")

# ---------------- why: SHAP for this one person ----------------
sv = explainer.shap_values(row)
if isinstance(sv, list):
    vals = sv[1][0]
elif sv.ndim == 3:
    vals = sv[0, :, 1]
else:
    vals = sv[0]
contrib = pd.Series(vals, index=features)

st.subheader("What pushed this estimate")
order = contrib.abs().sort_values().index
labels = []
for f in order:
    v = row.iloc[0][f]
    if f == "M/F":
        shown = sex
    elif f in ("nWBV", "ASF"):
        shown = f"{v:.3f}"
    else:
        shown = f"{v:.0f}"
    labels.append(f"{NAMES[f]} = {shown}")

fig, ax = plt.subplots(figsize=(6, 3.6))
colors = ["#a8442f" if contrib[f] > 0 else "#4a6a85" for f in order]
ax.barh(labels, [contrib[f] for f in order], color=colors)
ax.axvline(0, color="#555555", linewidth=0.8)
ax.set_xlabel("Towards Nondemented   |   Towards Demented")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
st.pyplot(fig)

img_push = contrib[imaging].sum()
clin_push = contrib[clinical].sum()


def direction(x):
    return "towards demented" if x > 0 else "towards nondemented"


st.write(
    f"For this person, the brain scan measurements together pushed the "
    f"estimate **{direction(img_push)}** ({img_push:+.2f}), and the clinical "
    f"details pushed it **{direction(clin_push)}** ({clin_push:+.2f})."
)

# ---------------- about ----------------
with st.expander("About the model and how it was tested"):
    st.write(
        "Random Forest trained on the OASIS-2 longitudinal data (150 people, "
        "373 visits). The label is demented (including people who converted "
        "during the study) versus nondemented. Brain scan measurements are "
        "eTIV, nWBV and ASF. Clinical details are age, sex, SES and MMSE."
    )
    s1 = Path("final_model_summary.csv")
    if s1.exists():
        st.write("Cross-validation on OASIS-2 (visits of one person kept "
                 "together):")
        st.dataframe(pd.read_csv(s1), hide_index=True)
    s2 = Path("oasis1_test_results.csv")
    if s2.exists():
        st.write("Trained on OASIS-2, tested on OASIS-1:")
        st.dataframe(pd.read_csv(s2), hide_index=True)
    st.write(
        "Limits: small dataset, people aged 60 and above only, and MMSE is "
        "closely linked to how the dementia label is assigned, so it has the "
        "biggest influence. The explanation chart shows what the model "
        "relied on, not what causes dementia."
    )
