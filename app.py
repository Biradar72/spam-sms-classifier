import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load the pre-trained scaler and model
scaler = pickle.load(open("scaler.pkl", "rb"))
model = pickle.load(open("log_model.pkl", "rb"))

st.title("Chronic Kidney Disease Prediction App")

st.write("""
This app predicts the **Possibility of Chronic Kidney Disease (CKD)** based on user inputs!
""")

st.sidebar.header("User Input Features")

# Function to get manual user input
def user_input_features():
    age = st.sidebar.slider("Age", 2, 90, 50)
    bp = st.sidebar.slider("Blood Pressure", 50, 180, 80)
    sg = st.sidebar.slider("Specific Gravity", 1.0, 1.025, 1.01)
    al = st.sidebar.slider("Albumin", 0, 5, 0)
    su = st.sidebar.slider("Sugar", 0, 5, 0)
    rbc = st.sidebar.selectbox("Red Blood Cells (Normal=0, Abnormal=1)", (0, 1))
    pc = st.sidebar.selectbox("Pus Cell (Normal=0, Abnormal=1)", (0, 1))
    pcc = st.sidebar.selectbox("Pus Cell Clumps (Not Present=0, Present=1)", (0, 1))
    ba = st.sidebar.selectbox("Bacteria (Not Present=0, Present=1)", (0, 1))
    bu = st.sidebar.slider("Blood Urea", 1, 400, 50)
    sc = st.sidebar.slider("Serum Creatinine", 0.1, 15.0, 1.2)
    sod = st.sidebar.slider("Sodium", 100, 200, 140)
    pot = st.sidebar.slider("Potassium", 2, 10, 4)
    hemo = st.sidebar.slider("Hemoglobin", 3, 17, 13)
    wc = st.sidebar.slider("White Blood Cell Count", 2000, 30000, 8000)
    htn = st.sidebar.selectbox("Hypertension (No=0, Yes=1)", (0, 1))
    dm = st.sidebar.selectbox("Diabetes Mellitus (No=0, Yes=1)", (0, 1))
    cad = st.sidebar.selectbox("Coronary Artery Disease (No=0, Yes=1)", (0, 1))
    appet = st.sidebar.selectbox("Appetite (Good=1, Poor=0)", (0, 1))
    pe = st.sidebar.selectbox("Pedal Edema (No=0, Yes=1)", (0, 1))
    ane = st.sidebar.selectbox("Anemia (No=0, Yes=1)", (0, 1))

    data = {
        'age': age, 'bp': bp, 'sg': sg, 'al': al, 'su': su, 'rbc': rbc, 'pc': pc, 
        'pcc': pcc, 'ba': ba, 'bu': bu, 'sc': sc, 'sod': sod, 'pot': pot, 'hemo': hemo, 
        'wc': wc, 'htn': htn, 'dm': dm, 'cad': cad, 'appet': appet, 'pe': pe, 'ane': ane
    }

    features = pd.DataFrame(data, index=[0])
    return features

# Upload CSV or manual input
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    st.sidebar.success("CSV uploaded successfully!")
else:
    input_df = user_input_features()

# Prediction
if st.sidebar.button("Predict"):
    data_scaled = scaler.transform(input_df)
    predictions = model.predict(data_scaled)
    predictions_proba = model.predict_proba(data_scaled)

    st.subheader("User Input Features")
    st.write(input_df)

    st.subheader("Prediction Result")
    result = "CKD Detected" if predictions[0] == 1 else "No CKD"
    st.write(f"### 🩺 {result}")

    st.subheader("Prediction Probability")
    st.write(f"No CKD: {predictions_proba[0][0]:.2f}")
    st.write(f"CKD: {predictions_proba[0][1]:.2f}")

    # Graph: Probability Bar Chart
    st.subheader("Probability Graph")
    fig, ax = plt.subplots()
    labels = ["No CKD", "CKD"]
    ax.bar(labels, predictions_proba[0], color=["green", "red"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title("CKD Prediction Probability")
    st.pyplot(fig)
