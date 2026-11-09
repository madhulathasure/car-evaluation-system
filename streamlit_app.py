import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import streamlit as st
from src.predict import CarEvaluator

st.set_page_config(page_title="Car Evaluation System", page_icon="🚗")
st.title("🚗 Car Evaluation System")
st.write("Select car attributes to predict its acceptability class.")

@st.cache_resource
def load():
    return CarEvaluator()

evaluator = load()

col1, col2 = st.columns(2)

with col1:
    buying = st.selectbox("Buying Price", ["vhigh", "high", "med", "low"])
    maint = st.selectbox("Maintenance Cost", ["vhigh", "high", "med", "low"])
    doors = st.selectbox("Number of Doors", ["2", "3", "4", "5"])

with col2:
    persons = st.selectbox("Passenger Capacity", ["2", "4", "6"],
                           format_func=lambda x: {"2": "2", "4": "4", "6": "more"}[x])
    lug_boot = st.selectbox("Luggage Boot Size", ["small", "med", "big"])
    safety = st.selectbox("Safety Rating", ["low", "med", "high"])

if st.button("Evaluate Car"):
    row = {"buying": buying, "maint": maint, "doors": doors,
           "persons": persons, "lug_boot": lug_boot, "safety": safety}
    label, proba = evaluator.predict(row)

    st.subheader(f"Prediction: **{label.upper()}**")
    if proba:
        st.write("Class probabilities:")
        for cls, p in sorted(proba.items(), key=lambda x: x[1], reverse=True):
            st.write(f"{cls} — {p}")
            st.progress(min(p, 1.0))