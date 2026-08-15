import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Salary Level Predictor", page_icon="💰", layout="wide")

# App Header
st.markdown("<h1 style='text-align: center; color: #1E88E5;'>💰 Adult Census Income Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict whether an individual earns more than $50,000 / year</p>", unsafe_allow_html=True)

# Load model & feature list using joblib
@st.cache_resource
def load_artifacts():
    model = joblib.load('model.joblib')
    feature_columns = joblib.load('model_features.joblib')
    return model, feature_columns

try:
    model, feature_columns = load_artifacts()
except Exception as e:
    st.error("⚠️ Could not load `model.joblib` or `model_features.joblib`. Make sure you generated them first!")
    st.stop()

# Sidebar User Inputs
st.sidebar.header("📋 Input Profile")

age = st.sidebar.slider("Age", 17, 90, 35)
workclass = st.sidebar.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])
education = st.sidebar.selectbox("Education Level", ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'])

edu_map = {'Preschool': 1, '1st-4th': 2, '5th-6th': 3, '7th-8th': 4, '9th': 5, '10th': 6, '11th': 7, '12th': 8, 'HS-grad': 9, 'Some-college': 10, 'Assoc-voc': 11, 'Assoc-acdm': 12, 'Bachelors': 13, 'Masters': 14, 'Prof-school': 15, 'Doctorate': 16}
education_num = edu_map.get(education, 9)

marital_status = st.sidebar.selectbox("Marital Status", ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
occupation = st.sidebar.selectbox("Occupation", ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'])
relationship = st.sidebar.selectbox("Relationship", ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'])
race = st.sidebar.selectbox("Race", ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'])
sex = st.sidebar.radio("Sex", ['Male', 'Female'])

col1, col2 = st.sidebar.columns(2)
with col1:
    capital_gain = st.number_input("Capital Gain ($)", 0, 100000, 0, 1000)
with col2:
    capital_loss = st.number_input("Capital Loss ($)", 0, 5000, 0, 100)

hours_per_week = st.sidebar.slider("Hours per Week", 1, 99, 40)
native_country = st.sidebar.selectbox("Native Country", ['United-States', 'Mexico', 'Philippines', 'Germany', 'Canada', 'Other'])

input_dict = {
    'age': age, 'workclass': workclass, 'fnlwgt': 178356, 'education': education,
    'education-num': education_num, 'marital-status': marital_status, 'occupation': occupation,
    'relationship': relationship, 'race': race, 'sex': sex, 'capital-gain': capital_gain,
    'capital-loss': capital_loss, 'hours-per-week': hours_per_week, 'native-country': native_country
}

input_df = pd.DataFrame([input_dict])

# Main Display
c1, c2 = st.columns([1, 1.2])

with c1:
    st.subheader("👤 Selected Profile Summary")
    m1, m2, m3 = st.columns(3)
    m1.metric("Age", f"{age} yrs")
    m2.metric("Edu Num", f"Lvl {education_num}")
    m3.metric("Work Hrs", f"{hours_per_week}h/wk")
    st.dataframe(input_df.T.rename(columns={0: "Selection"}), use_container_width=True)

with c2:
    st.subheader("📊 Model Prediction")
    encoded_input = pd.get_dummies(input_df).reindex(columns=feature_columns, fill_value=0)

    if st.button("🚀 Predict Income Class"):
        pred = model.predict(encoded_input)[0]
        probs = model.predict_proba(encoded_input)[0]

        prob_low, prob_high = probs[0] * 100, probs[1] * 100

        st.write("---")
        if pred == 1:
            st.success(f"### 🎉 Result: > $50K / year")
            st.metric("Model Confidence", f"{prob_high:.1f}%")
        else:
            st.info(f"### ℹ️ Result: <= $50K / year")
            st.metric("Model Confidence", f"{prob_low:.1f}%")

        fig, ax = plt.subplots(figsize=(6, 2.2))
        sns.barplot(x=[prob_low, prob_high], y=['<= $50K', '> $50K'], palette=['#64B5F6', '#4CAF50'], ax=ax)
        ax.set_xlim(0, 100)
        for p in ax.patches:
            ax.annotate(f'{p.get_width():.1f}%', (p.get_width() + 2, p.get_y() + p.get_height() / 2), va='center', fontweight='bold')
        st.pyplot(fig)
