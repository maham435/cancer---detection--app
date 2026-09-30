import streamlit as st
import pickle
import numpy as np

# Trained Model ko load karain
try:
    model = pickle.load(open('cancer_model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Error: 'cancer_model.pkl' file nahi mili!")

st.set_page_config(page_title="Cancer Detection App", layout="centered")
st.title("🎗️ Breast Cancer Detection System")
st.write("Machine Learning ki madad se tumor ka pata lagayein.")
st.markdown("---")

radius_mean = st.number_input("Radius Mean (Tumor ka size):", min_value=0.0, value=14.0)
texture_mean = st.number_input("Texture Mean (Sath ki khurdrahat):", min_value=0.0, value=20.0)

if st.button("🔎 Predict Cancer Status", use_container_width=True):
    input_features = [radius_mean, texture_mean] + [0.0]*28
    prediction = model.predict([input_features])
    
    st.subheader("📋 Detection Result:")
    if prediction == 1:
        st.error("⚠️ **Result: Malignant (Cancerous)**\n\nTumor mein cancer ke aasaar hain. Baraye meherbani doctor se jald rabhta karain.")
    else:
        st.success("✅ **Result: Benign (Non-Cancerous)**\n\nTumor normal hai, cancer ke koi aasaar nahi hain.")
