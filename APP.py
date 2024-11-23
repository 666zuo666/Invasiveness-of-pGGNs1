import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the fetal state model
model = joblib.load('LightGBM.pkl')

# Define feature names
feature_names = [
    "ITH score", "Lesion size", "Age", "CT value", "Spiculation sign", "Pleural indentation sign", "Location", "Vascular convergence sign", "Lobulation sign",
    "Vacuole sign", "Sex", "Margin"]

# Streamlit user interface
st.title("Invasion Predictor")

# Input features
ITH_score = st.number_input("ITH score:", min_value=0.0, max_value=1.0, value=0.41, step=0.01)
Lesion_size= st.number_input("Lesion size:", min_value=1, max_value=30, value=12)
Age= st.number_input("Age:", min_value=25, max_value=90, value=52)
CT_value = st.number_input("CT value:", min_value=-800, max_value=-220, value=-320)
Spiculation_sign= st.selectbox("Spiculation sign:", options=[0, 1], format_func=lambda x: "Absent" if x == 0 else "Present")
Pleural_indentation_sign= st.selectbox("Pleural indentation sign:", options=[0, 1], format_func=lambda x: "Absent" if x == 0 else "Present")
Location = st.selectbox("Location:", options=[1, 2, 3, 4, 5], format_func=lambda x: "RUL" if x == 1 else ("RLL" if x == 2 else ("RML" if x == 3 else ("LUL" if x == 4 else "LLL"))))
Vascular_convergence_sign= st.selectbox("Vascular convergence sign:", options=[0, 1], format_func=lambda x: "Absent" if x == 0 else "Present")
Lobulation_sign= st.selectbox("Lobulation sign:", options=[0, 1], format_func=lambda x: "Absent" if x == 0 else "Present")
Vacuole_sign= st.selectbox("Vacuole sign:", options=[0, 1], format_func=lambda x: "Absent" if x == 0 else "Present")
Sex= st.selectbox("Sex:", options=[2, 1], format_func=lambda x: "Male" if x == 1 else "Female")
Margin= st.selectbox("Margin:", options=[0, 1], format_func=lambda x: "Absent" if x == 0 else "Present")

# Collect input values into a list
feature_values = [ITH_score, Lesion_size,Age, CT_value , Spiculation_sign, Pleural_indentation_sign, Location, Vascular_convergence_sign,Lobulation_sign,Vacuole_sign, Sex, Margin]

# Convert feature values into a DataFrame
features_df = pd.DataFrame([feature_values], columns=feature_names)

if st.button("Predict"):
    # Predict class and probabilities using DataFrame
    predicted_class = model.predict(features_df)[0]
    predicted_proba = model.predict_proba(features_df)[0]

    # Display prediction results
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 0:
        advice = (
            f"Our model indicates high probability of pGGNs being pathologically identified as AIS. "
            f"The model estimates the probability of AIS as {probability:.1f}%. "
            "It's important to maintain a healthy lifestyle and keep having regular check-ups."
        )
    elif predicted_class == 1:
        advice = (
            f"Our model indicates high probability of pGGNs being pathologically identified as MIA.  "
            f"The model estimates the probability of MIA as {probability:.1f}%. "
            "Further evaluation and close monitoring are recommended."
        )
    else:
        advice = (
            f"Our model indicates high probability of pGGNs being pathologically identified as IAC. "
            f"The model estimates the probability of IAC as {probability:.1f}%.  "
            "Operative intervention is advised, specifically an anatomic lobectomy in conjunction with systematic lymph node dissection."
        )

    st.write(advice)

    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values_Explanation = explainer(features_df)

    # Display SHAP waterfall plot only for the predicted class
    plt.figure(figsize=(10, 5), dpi=1200)
    shap.plots.waterfall(shap_values_Explanation[:,:,predicted_class][0], show=False, max_display=13)
    plt.savefig("shap_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_plot.png")
