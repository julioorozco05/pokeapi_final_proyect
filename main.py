from fastapi import FastAPI, status, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import joblib

app = FastAPI(
    title="Pokémon Type Prediction",
    version="0.0.2"
)

# Cargar el modelo entrenado
model = joblib.load("model/logistic_regression_tipo_pokemon.pkl")


@app.post("/api/v1/predict-pokemon-type", tags=["pokemon-type-prediction"])
async def predict(
        hp: float,
        attack: float,
        defense: float,
        sp_attack: float,
        sp_defense: float,
        speed: float,
        generation: float
):
    try:
        # Crear el DataFrame directamente desde el diccionario
        data_frame = pd.DataFrame({
            'hp': [hp],
            'attack': [attack],
            'defense': [defense],
            'sp_attack': [sp_attack],
            'sp_defense': [sp_defense],
            'speed': [speed],
            'generation': [generation]
        })

        # Realizar la predicción
        prediction = model.predict(data_frame)
        predicted_type = prediction[0]  # Suponiendo que la predicción es un número que representa el tipo

        # Devolver la predicción como JSON
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"predicted_type": predicted_type}  # Devolver el tipo predicho dentro de un diccionario
        )
    except Exception as e:
        # Manejar los errores
        raise HTTPException(
            detail=str(e),
            status_code=status.HTTP_400_BAD_REQUEST
        )












