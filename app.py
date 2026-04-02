import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Title
st.set_page_config(page_title="Asthma Prediction", layout="centered")

st.title("Asthma Risk Prediction System 🌫️")
st.markdown("### Analyze impact of air pollution on asthma")

st.write("Enter pollution levels to predict asthma risk")

st.sidebar.header("Enter Pollution Levels")

pm25 = st.sidebar.slider("PM2.5", 0, 300, 50)
no2 = st.sidebar.slider("NO2", 0, 150, 30)
co = st.sidebar.slider("CO", 0.0, 5.0, 1.0)


# Predict button
if st.sidebar.button("Predict"):
    input_data = np.array([[pm25, no2, co]])
    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)

    st.subheader("Prediction Result")
    st.success(f"Predicted Asthma Risk: {prediction[0]}")
    st.write("Confidence Levels:")
    st.write(prob)


data = pd.read_csv("dataset.csv")

st.subheader("📊 Data Visualization")
fig, ax = plt.subplots()
ax.scatter(data['PM25'], data['NO2'])
ax.set_xlabel("PM2.5")
ax.set_ylabel("NO2")
ax.set_title("Pollution Distribution")
st.pyplot(fig)
st.subheader("📖 About Project")

st.write("""
This system predicts asthma risk based on air pollution levels using Machine Learning.
Higher pollution levels increase asthma risk.
""")

st.subheader("📊 Asthma Risk Distribution")

risk_counts = data['Asthma_Risk'].value_counts()
st.bar_chart(risk_counts)

st.subheader("📄 Dataset Preview")
st.dataframe(data)

st.subheader("🧠 Interpretation")

if pm25 > 150:
    st.warning("Very high pollution! Severe asthma risk.")
elif pm25 > 80:
    st.info("Moderate pollution. Be cautious.")
else:
    st.success("Air quality is relatively safe.")

st.subheader("✨ Features")

st.write("""
- Predict asthma risk using ML
- Real-time pollution input
- Data visualization
- User-friendly interface
""")