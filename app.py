import streamlit as st
import joblib

# Load models
tfidf = joblib.load("tfidf.pkl")
classifier = joblib.load("classifier.pkl")
regressor = joblib.load("regressor.pkl")

st.set_page_config(page_title="AutoJudge", layout="centered")

st.title("🧠 AutoJudge")
st.subheader("Predict Programming Problem Difficulty")

st.write("Paste the problem details below:")

description = st.text_area("Problem Description")
input_desc = st.text_area("Input Description")
output_desc = st.text_area("Output Description")

if st.button("Predict"):
    combined_text = description + " " + input_desc + " " + output_desc
    features = tfidf.transform([combined_text])

    predicted_class = classifier.predict(features)[0]
    predicted_score = regressor.predict(features)[0]

    st.success(f"📌 Predicted Difficulty Class: **{predicted_class}**")
    st.success(f"📊 Predicted Difficulty Score: **{round(predicted_score, 2)}**")
