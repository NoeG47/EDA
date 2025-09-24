import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def train_discount_model(df):
    """
    Entrena un modelo de regresión para predecir el descuento.
    """
    # Limpiar los nombres de las columnas
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # Definir las variables
    features = [
        "product_name",
        "category",
        "price",
        "units_sold",
        "location",
        "platform",
    ]
    X = df[features]
    y = df["discount"]

    # Preprocesamiento: Codificación para variables categóricas
    categorical_features = ["product_name", "category", "location", "platform"]
    numerical_features = ["price", "units_sold"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
            ("num", "passthrough", numerical_features),
        ]
    )

    # Crear el pipeline y entrenar el modelo
    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=100, random_state=42)),
        ]
    )

    model_pipeline.fit(X, y)

    return model_pipeline


def predict_discount_for_product(
    model, product, category, price, units_sold, location, platform
):
    """
    Predice el descuento para un producto específico.
    """
    # Crear un DataFrame con los datos de entrada
    input_data = pd.DataFrame(
        [
            {
                "product_name": product,
                "category": category,
                "price": price,
                "units_sold": units_sold,
                "location": location,
                "platform": platform,
            }
        ]
    )

    # Predecir el descuento
    predicted_discount = model.predict(input_data)[0]

    # Asegurarse de que el descuento no sea negativo
    return max(0, predicted_discount)
