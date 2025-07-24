# streamlit_ui.py — Streamlit Frontend

import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import base64

# Constants
API_URL = "http://localhost:8000/predict"  # Change this if deployed elsewhere
symptom_list = joblib.load("all_symptoms.pkl")
desc_df = pd.read_csv("symptom_Description.csv")
prec_df = pd.read_csv("symptom_precaution.csv")
main_df = pd.read_csv("dataset.csv")

display_name_map = {sym.replace("_", " ").title(): sym for sym in symptom_list}
display_names = list(display_name_map.keys())

st.set_page_config(page_title="Smart Symptom Classifier", page_icon="⚕", layout="centered")
tab1, tab2, tab3 = st.tabs(["➤ Diagnose", "➤ Dashboard", "➤ About & Notes"])

def add_bg_from_local(image_file):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
    """, unsafe_allow_html=True)

add_bg_from_local("background.jpg")

with tab1:
    st.title("🩺 Smart Symptom Classifier")
    st.markdown("### ➥ Diagnose common diseases based on your selected symptoms")
    st.markdown("**Disclaimer!** This tool is not intended for real-world clinical use.")

    selected_display = st.multiselect("Select your symptoms:", options=display_names)
    selected_symptoms = [display_name_map[name] for name in selected_display]

    if selected_symptoms:
        response = requests.post(API_URL, json={"symptoms": selected_symptoms})
        if response.status_code == 200:
            data = response.json()

            # Focus Disease Risks
            st.markdown("### 🧬 Risk Assessment for Key Diseases")
            for entry in data["focus_diseases"]:
                disease = entry["disease"]
                conf = entry["confidence"]
                risk = entry["risk"]
                color = "🟥" if risk == "High" else "🟨" if risk == "Medium" else "🟩"
                st.write(f"{color} **{disease}** — Confidence: `{conf}` | Risk Level: **{risk}**")

            # Top Predictions
            st.markdown("### 🔍 Overall Top 3 Predicted Diseases")
            for i, entry in enumerate(data["top_diseases"]):
                prefix = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                st.info(f"{prefix} {entry['disease']} — `{entry['confidence']}` confidence")

            # Description + Precaution (Top disease only)
            top_disease = data["top_diseases"][0]["disease"]
            desc_row = desc_df[desc_df["Disease"] == top_disease]
            if not desc_row.empty:
                st.markdown(f"✎ **What is it?** {desc_row['Description'].values[0]}")
            prec_row = prec_df[prec_df["Disease"] == top_disease]
            if not prec_row.empty:
                st.markdown("💊 **Recommended Precautions:**")
                for i in range(1, 5):
                    val = prec_row[f"Precaution_{i}"].values[0]
                    if pd.notna(val):
                        st.write(f"- {val}")
        else:
            st.error("Failed to get prediction from server. Please check the backend.")

with tab2:
    st.title("📊 Dataset Dashboard")
    disease_counts = main_df["Disease"].value_counts()
    st.subheader("➥ Number of Samples per Disease")
    st.bar_chart(disease_counts)

    st.subheader("➥ Most Common Symptoms")
    all_symptoms_flat = main_df[[col for col in main_df.columns if col.startswith("Symptom")]].values.flatten()
    all_symptoms_flat = pd.Series(all_symptoms_flat).dropna().str.replace(" ", "_").str.lower()
    wordcloud = WordCloud(width=1000, height=400, background_color="white").generate(" ".join(all_symptoms_flat))

    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

    if st.checkbox("➥ Show raw dataset"):
        st.write(main_df.head(20))

with tab3:
    st.title("📝 About This Project & Learnings")
    st.markdown("""... (same About text as before) ...""")

st.markdown("""<hr style="margin-top: 3em; margin-bottom: 1em;"/>""", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #aaa; font-size: 14px;'>
Made with ❤️ by <b>Zain Hassan Gondal</b><br>
<a href='https://www.linkedin.com/in/zain-hassan-gondal' target='_blank'>LinkedIn</a> &nbsp;|&nbsp;
<a href='https://github.com/zainhassanee/Multiple-Disease-Prediction-Ai-' target='_blank'>GitHub</a>
</div>
""", unsafe_allow_html=True)
