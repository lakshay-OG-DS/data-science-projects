import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(
    page_title="Genshin Impact Rarity Predictor", layout="wide"
)

st.markdown(
    "<h1 style='text-align: center; color: #9C27B0;'>Genshin Impact"
    " Character Rarity Predictor</h1>",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifact():
  return joblib.load("impact.pkl")


try:
  artifact = load_artifact()
  model = artifact["model"]
  feature_columns = artifact["features"]
except Exception as e:
  st.error("Could not load impact.pkl. Make sure it is in your folder!")
  st.stop()

st.sidebar.header("Character Attributes")
element = st.sidebar.selectbox(
    "Element", ["Pyro", "Hydro", "Electro", "Cryo", "Anemo", "Geo", "Dendro"]
)
region = st.sidebar.selectbox(
    "Region",
    ["Mondstadt", "Liyue", "Inazuma", "Sumeru", "Fontaine", "Natlan", "Snezhnaya"],
)
roles = st.sidebar.selectbox(
    "Primary Role", ["DPS", "Sub-DPS", "Support", "Healer", "Shielder"]
)

col1, col2 = st.sidebar.columns(2)
with col1:
  lvl_90_atk = st.number_input("Level 90 ATK", 100, 400, 250)
with col2:
  lvl_90_def = st.number_input("Level 90 DEF", 300, 1000, 700)

lvl_90_hp = st.number_input("Level 90 HP", 8000, 20000, 12000)
num_banners = st.slider("Number of Banners", 1, 20, 3)
is_archon = st.sidebar.selectbox("Is Archon?", [0, 1])

input_dict = {
    "element": element,
    "region": region,
    "roles": roles,
    "lvl_90_ATK": lvl_90_atk,
    "lvl_90_DEF": lvl_90_def,
    "lvl_90_HP": lvl_90_hp,
    "num_banners": num_banners,
    "is_archon": is_archon,
}

input_df = pd.DataFrame([input_dict])

c1, c2 = st.columns([1, 1.2])

with c1:
  st.subheader("Profile Summary")
  st.dataframe(
      input_df.T.rename(columns={0: "Selection"}), use_container_width=True
  )

with c2:
  st.subheader("Model Prediction")
  encoded_input = pd.get_dummies(input_df).reindex(
      columns=feature_columns, fill_value=0
  )

  if st.button("Predict Character Rarity", use_container_width=True):
    pred = model.predict(encoded_input)[0]
    probs = model.predict_proba(encoded_input)[0]
    classes = model.classes_
    prob_dict = {cls: prob * 100 for cls, prob in zip(classes, probs)}

    if pred == 5:
      st.success("Result: 5-Star Character")
      st.metric("Model Confidence", f"{prob_dict.get(5, 0.0):.1f}%")
    else:
      st.info("Result: 4-Star Character")
      st.metric("Model Confidence", f"{prob_dict.get(4, 0.0):.1f}%")
      