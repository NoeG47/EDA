import streamlit as st
import pandas as pd
import numpy as np
from model_utils import train_discount_model, predict_discount_for_product


# --- Cargar datos y entrenar el modelo ---
@st.cache_data
def load_data_and_train_model():
    df = pd.read_csv("data/sales_data.csv")
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    for col in ["product_name", "category", "location", "platform"]:
        if col in df.columns:
            df[col] = df[col].str.strip()

    model = train_discount_model(df)
    return df, model


df, model = load_data_and_train_model()

# --- Configuración de la interfaz de usuario ---
st.set_page_config(
    page_title="Predictor de Descuento Realista",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎯 Predictor de Descuento Realista")
st.markdown(
    "Usa este modelo para predecir un descuento 'realista' para un producto basado en tu historial de ventas."
)

# --- Sidebar para las entradas del usuario ---
st.sidebar.header("Selecciona las variables del producto")

# Paso 1: Selecciona el producto primero
product_list = sorted(df["product_name"].unique())
selected_product = st.sidebar.selectbox("Producto", product_list)

# Paso 2: Filtra las categorías basándote en el producto seleccionado
filtered_categories = df[df["product_name"] == selected_product]["category"].unique()
selected_category = st.sidebar.selectbox("Categoría", filtered_categories)

# El resto del código permanece igual

location_list = sorted(df["location"].unique())
platform_list = sorted(df["platform"].unique())

selected_location = st.sidebar.selectbox("Ubicación", location_list)
selected_platform = st.sidebar.selectbox("Plataforma", platform_list)

# Manejar los valores por defecto para el precio y unidades vendidas
default_price = float(df[df["product_name"] == selected_product]["price"].mean())
selected_price = st.sidebar.number_input(
    "Precio por unidad", min_value=1.0, value=default_price, step=0.1
)

default_units = int(df[df["product_name"] == selected_product]["units_sold"].mean())
selected_units_sold = st.sidebar.number_input(
    "Unidades Vendidas (históricas)", min_value=1, value=default_units
)


# --- Lógica de predicción ---
if st.sidebar.button("Predecir Descuento"):
    predicted_discount = predict_discount_for_product(
        model,
        selected_product,
        selected_category,
        selected_price,
        selected_units_sold,
        selected_location,
        selected_platform,
    )

    st.subheader("Resultado de la Predicción")

    # Mostrar el resultado clave
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label="Descuento Realista Predicho", value=f"{predicted_discount*100:.2f}%"
        )
    with col2:
        st.metric(label="Valor del Descuento", value=f"{predicted_discount:.4f}")
