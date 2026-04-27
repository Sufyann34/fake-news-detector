import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Fake News Detector", page_icon="🔍", layout="centered")

st.title("🔍 Fake News Detector")
st.caption("Paste a news article or headline — the model will classify it instantly.")

text = st.text_area("News text", placeholder="Paste your article or headline here…", height=180)

if st.button("Analyse", type="primary"):
    if len(text.strip()) < 10:
        st.warning("Please enter at least 10 characters.")
    else:
        with st.spinner("Analysing…"):
            resp = requests.post(f"{API_URL}/predict", json={"text": text})
            data = resp.json()

        label      = data["label"].upper()
        confidence = data["confidence"]
        latency    = data["latency_ms"]

        if label == "FAKE":
            st.error(f"🚨 {label} NEWS — {confidence*100:.1f}% confidence")
        else:
            st.success(f"✅ REAL NEWS — {confidence*100:.1f}% confidence")

        st.progress(confidence)
        st.caption(f"Inference latency: {latency} ms")