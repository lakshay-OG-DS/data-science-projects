import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(
    page_title="Genshin Impact Rarity Predictor", layout="wide"
)

# App Header
st.markdown(
    "<h1 style='text-align: center; color: #9C27B0;'>Genshin Impact"
    " Character Rarity Predictor</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center;'>Predict whether a character is a 4-Star or"
    " 5-Star unit based on attributes and baseline level 90 stats</p>",
    unsafe_allow_html=True,
)


# Load model & feature list using joblib (.pkl files)
@st.cache_resource
def load_artifacts():
  model = joblib.load("genshin_model.pkl")
  feature_columns = joblib.load("genshin_model_features.pkl")
  return model, feature_columns


try:
  model, feature_columns = load_artifacts()
except Exception as e:
  st.error(
      "Could not load model files (`genshin_model.pkl` or"
      " `genshin_model_features.pkl`). Make sure you ran `genshin.py` first!"
  )
  st.stop()

# Sidebar User Inputs
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

# Main Display Layout
c1, c2 = st.columns([1, 1.2])

with c1:
  st.subheader("Profile Summary")
  m1, m2, m3 = st.columns(3)
  m1.metric("Atk", f"{lvl_90_atk}")
  m2.metric("Banners", f"{num_banners}")
  m3.metric("Archon", "Yes" if is_archon == 1 else "No")
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

    prob_4star = prob_dict.get(4, 0.0)
    prob_5star = prob_dict.get(5, 0.0)

    st.write("---")
    if pred == 5:
      st.success("Result: 5-Star Character")
      st.metric("Model Confidence", f"{prob_5star:.1f}%")
    else:
      st.info("Result: 4-Star Character")
      st.metric("Model Confidence", f"{prob_4star:.1f}%")

    fig, ax = plt.subplots(figsize=(6, 2.2))
    sns.barplot(
        x=[prob_4star, prob_5star],
        y=["4-Star", "5-Star"],
        palette=["#42A5F5", "#FFCA28"],
        ax=ax,
    )
    ax.set_xlim(0, 100)
    for p in ax.patches:
      ax.annotate(
          f"{p.get_width():.1f}%",
          (p.get_width() + 2, p.get_y() + p.get_height() / 2),
          va="center",
          fontweight="bold",
      )
    st.pyplot(fig)