
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import joblib
import json
import random

app = FastAPI(
    title="Pokémon Type Prediction API",
    version="0.0.2"
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Cargar el modelo entrenado
model = joblib.load("model/logistic_regression_tipo_pokemon.pkl")

# Cargar los datos de los Pokémon desde un archivo JSON
with open("data.json", "r") as file:
    pokemon_data = json.load(file)

@app.post("/api/v1/predict-pokemon-type", tags=["pokemon-type-prediction"])
async def predict_pokemon_type(
    hp: float,
    attack: float,
    defense: float,
    sp_attack: float,
    sp_defense: float,
    speed: float,
    generation: float
):
    try:
        # Crear el DataFrame directamente desde los parámetros recibidos
        data_frame = pd.DataFrame({
            'hp': [hp],
            'attack': [attack],
            'defense': [defense],
            'sp_attack': [sp_attack],
            'sp_defense': [sp_defense],
            'speed': [speed],
            'generation': [generation]
        })

        # Realizar la predicción del tipo de Pokémon
        prediction = model.predict(data_frame)
        predicted_type = prediction[0]

        # Buscar la imagen asociada al tipo predicho en el archivo JSON
        predicted_pokemon = next((pokemon for pokemon in pokemon_data if pokemon["predicted_type"] == predicted_type), None)
        if predicted_pokemon is None:
            raise HTTPException(
                detail=f"No se encontró una imagen asociada con el tipo de Pokémon predicho: {predicted_type}",
                status_code=status.HTTP_404_NOT_FOUND
            )

        # Construir la respuesta HTML con la imagen asociada al tipo predicho
        html_content = f"""
        <html>
        <head>
            <title>Pokémon Prediction</title>
            <link rel="stylesheet" href="/static/styles.css">
        </head>
        <body>
            <h1>Pokémon Type Prediction</h1>
            <p>The predicted type is: {predicted_type}</p>
            <img src="{predicted_pokemon['pokemon_image_url']}" alt="{predicted_type}">
        </body>
        </html>
        """

        return HTMLResponse(content=html_content, status_code=status.HTTP_200_OK)
    except Exception as e:
        raise HTTPException(
            detail=str(e),
            status_code=status.HTTP_400_BAD_REQUEST
        )










