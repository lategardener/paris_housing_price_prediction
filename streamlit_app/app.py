import os

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib   # if you want to load a pickled model

# ---------- 1. Load data (cached) ----------
@st.cache_data
def load_data():
    return pd.read_csv("data/paris_housing.csv")

def load_metrics():
    return pd.read_csv("outputs/metrics_outputs/model_metrics.csv")

df = load_data()

# ---------- 2. Sidebar: navigation ----------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Exploration", "Model Comparison", "Predictions"])

# ---------- 3. Exploration Page ----------
if page == "Exploration":
    st.header("Data Exploration")

    col = st.selectbox("Choose a numeric column", df.columns[:-1])

    st.subheader(f"{col} distribution")
    st.bar_chart(df[col])

    st.subheader(f"Relationship between {col} and price")
    fig = px.scatter(df, x=col, y="price")
    st.plotly_chart(fig)

# ---------- 4. Model Comparison Page ----------
elif page == "Model Comparison":
    st.header("Metrics of Different Models")

    # Put your real values here ↘
    metrics = load_metrics()

    st.dataframe(metrics, use_container_width=True)

    # Choix de la métrique
    metric = st.selectbox("Metric to display", ["MAE", "RMSE", "R2"])

    # Graphique interactif avec Plotly
    fig = px.bar(metrics, x=metric, y="Model", orientation="h", color="Model",
                 title=f"{metric} by Model", color_discrete_sequence=px.colors.qualitative.Vivid)

    st.plotly_chart(fig, use_container_width=True)

# ---------- 5. Predictions Page ----------
else:
    st.header("Test a Prediction")
    st.write("Enter the features of a house; get the predicted price.")

    # 1. Liste des modèles disponibles dans outputs/models/
    model_dir = "outputs/models"
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".joblib")]
    model_names = [f.replace(".joblib", "") for f in model_files]

    selected_model_name = st.selectbox("Select a model", model_names)
    selected_model_path = os.path.join(model_dir, selected_model_name + ".joblib")


    # 2. Génération dynamique des champs de saisie en fonction des colonnes
    input_values = []
    st.subheader("Enter the house features")

    feature_cols = [col for col in df.columns if col != "price"]

    # Affichage par groupe de 3 colonnes
    for i in range(0, len(feature_cols), 3):
        cols = st.columns(4)  # créer 3 colonnes Streamlit
        for j, col in enumerate(feature_cols[i:i+3]):
            dtype = df[col].dtype
            with cols[j]:  # placer chaque champ dans une colonne
                if dtype == "bool":
                    val = st.checkbox(f"{col}", value=False)
                elif pd.api.types.is_integer_dtype(dtype):
                    val = st.number_input(f"{col} (int)", value=int(df[col].mean()))
                elif pd.api.types.is_float_dtype(dtype):
                    val = st.number_input(f"{col} (float)", value=float(df[col].mean()))
                else:
                    st.warning(f"Skipping unsupported column type for {col}")
                    continue

                input_values.append(val)


    # 3. Bouton de prédiction
    if st.button("Predict"):
        try:
            model = joblib.load(selected_model_path)
            pred = model.predict([input_values])
            st.success(f"Estimated price: {pred[0]:,.0f} €")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
