import streamlit as st
import joblib
from scipy.sparse import hstack
from features import handcrafted_features

# Load models
tfidf = joblib.load("tfidf.pkl")
classifier = joblib.load("classifier.pkl")
regressor = joblib.load("regressor.pkl")

st.set_page_config(page_title="AutoJudge", layout="centered")

st.markdown(
    """
    ## 🧠 AutoJudge  
    Let’s see how difficult your coding problem is 👀  
    Paste the problem details below and hit **Predict**.
    """
)

# ---------- Session state ----------
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "data" not in st.session_state:
    st.session_state.data = {}

# ---------- INPUT FORM ----------
if not st.session_state.submitted:
    with st.form("problem_form"):
        desc = st.text_area("Problem Description", height=300)
        inp = st.text_area("Input Description", height=200)
        out = st.text_area("Output Description", height=200)

        submitted = st.form_submit_button("Predict")

        if submitted:
            st.session_state.data = {
                "desc": desc,
                "inp": inp,
                "out": out,
            }
            st.session_state.submitted = True
            st.rerun()

# ---------- AFTER SUBMIT (RELOAD-LIKE VIEW) ----------
else:
    desc = st.session_state.data["desc"]
    inp = st.session_state.data["inp"]
    out = st.session_state.data["out"]

    # Expanded, read-only display
    st.text_area("Problem Description", desc, height=max(300, len(desc)//2), disabled=True)
    st.text_area("Input Description", inp, height=max(200, len(inp)//2), disabled=True)
    st.text_area("Output Description", out, height=max(200, len(out)//2), disabled=True)

    combined_text = desc + " " + inp + " " + out
    X_text = tfidf.transform([combined_text])
    X_hand = handcrafted_features([combined_text])
    X_final = hstack([X_text, X_hand])

    predicted_class = classifier.predict(X_final)[0]
    predicted_score = regressor.predict(X_final)[0]

    st.markdown("---")
    st.markdown("### 📊 Result")

    st.markdown(
        f"""
        <div style="background-color:#1f2933;padding:16px;border-radius:10px;">
        <b>Difficulty:</b> {predicted_class.capitalize()}<br>
        <b>Score:</b> {predicted_score:.2f} / 10
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("Try another problem"):
        st.session_state.submitted = False
        st.session_state.data = {}
        st.rerun()
