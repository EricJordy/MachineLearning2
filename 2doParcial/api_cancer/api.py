"""
API de clasificación de tumores (Breast Cancer Wisconsin)
Proyecto académico de Machine Learning

Ejecutar desde la carpeta api_cancer:
    uvicorn api:app --reload
Documentación interactiva:
    http://127.0.0.1:8000/docs
"""

from pathlib import Path
import math
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


# 1. Cargar el modelo y el orden de las características
CARPETA = Path(__file__).resolve().parent

modelo = joblib.load(CARPETA / "modelo_cancer.pkl")
columnas = joblib.load(CARPETA / "columnas_modelo.pkl")

# En este conjunto de datos: 0 = maligno, 1 = benigno
NOMBRES = {0: "Maligno", 1: "Benigno"}


# 2. Reconstruir los mismos datos de prueba que se usaron en el notebook
cancer = load_breast_cancer(as_frame=True)
df_datos = cancer.data[columnas].copy()
objetivo = cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    df_datos,
    objetivo,
    test_size=0.20,
    random_state=42,
    stratify=objetivo,
)

# Métricas calculadas una sola vez al iniciar la API
y_pred_test = modelo.predict(X_test)
METRICAS = {
    "exactitud": round(float(accuracy_score(y_test, y_pred_test)), 4),
    "total_prueba": int(len(y_test)),
    "aciertos": int((y_test.to_numpy() == y_pred_test).sum()),
    "etiquetas": ["Maligno", "Benigno"],
    # filas = diagnóstico real, columnas = predicción
    "matriz_confusion": confusion_matrix(
        y_test, y_pred_test, labels=[0, 1]
    ).tolist(),
}

# Correlación de cada característica con el diagnóstico
correlaciones = (
    df_datos.assign(diagnostico=objetivo)
    .corr()["diagnostico"]
    .drop("diagnostico")
)
orden = correlaciones.abs().sort_values(ascending=False).index
IMPORTANCIA = [
    {
        "parametro": nombre,
        "correlacion": round(float(correlaciones[nombre]), 6),
        "absoluta": round(abs(float(correlaciones[nombre])), 6),
    }
    for nombre in orden
]


# 3. Crear la aplicación
app = FastAPI(
    title="API de clasificación de cáncer de mama",
    description="Proyecto académico de Machine Learning. "
    "Los resultados no constituyen un diagnóstico médico.",
    version="2.0",
)

# Permitir que el HTML consulte la API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# 4. Esquema de los datos de entrada
class DatosMuestra(BaseModel):
    parametros: dict[str, float] = Field(
        description="Las 30 características del modelo con su valor numérico"
    )


def a_registros(X: pd.DataFrame, y: pd.Series, limite: int):
    """Convierte filas del DataFrame en una lista apta para JSON."""
    registros = []
    for indice in X.index[:limite]:
        registros.append({
            "indice": int(indice),
            "valores": {c: float(X.at[indice, c]) for c in columnas},
            "diagnostico_real": NOMBRES[int(y.at[indice])],
        })
    return registros


# 5. Rutas
@app.get("/")
def inicio():
    return {
        "mensaje": "API funcionando correctamente",
        "total_parametros": len(columnas),
        "documentacion": "/docs",
    }


@app.get("/parametros")
def obtener_parametros():
    return {"parametros": columnas}


@app.get("/importancia")
def obtener_importancia(limite: int = Query(10, ge=1, le=30)):
    return {"importancia": IMPORTANCIA[:limite]}


@app.get("/metricas")
def obtener_metricas():
    return METRICAS


@app.post("/predecir")
def predecir(datos: DatosMuestra):
    entrada = datos.parametros

    faltantes = set(columnas) - set(entrada)
    adicionales = set(entrada) - set(columnas)
    if faltantes or adicionales:
        raise HTTPException(
            status_code=422,
            detail={
                "parametros_faltantes": sorted(faltantes),
                "parametros_adicionales": sorted(adicionales),
            },
        )

    if not all(math.isfinite(v) for v in entrada.values()):
        raise HTTPException(
            status_code=422,
            detail="Los valores deben ser números finitos.",
        )

    # El DataFrame debe respetar el orden con el que se entrenó el modelo
    X = pd.DataFrame([[entrada[c] for c in columnas]], columns=columnas)

    prediccion = int(modelo.predict(X)[0])
    probabilidades = modelo.predict_proba(X)[0]

    return {
        "prediccion": prediccion,
        "resultado": NOMBRES[prediccion],
        "probabilidades_estimadas": {
            NOMBRES[int(clase)]: round(float(p), 4)
            for clase, p in zip(modelo.classes_, probabilidades)
        },
        "advertencia": "Resultado experimental. No constituye un diagnóstico médico.",
    }


@app.get("/registros-prueba")
def obtener_registros_prueba(limite: int = Query(20, ge=1, le=200)):
    """Muestras que el modelo NO vio durante el entrenamiento."""
    return {
        "total": int(len(X_test)),
        "registros": a_registros(X_test, y_test, limite),
    }


@app.get("/registros-reales")
def obtener_registros_reales(limite: int = Query(20, ge=1, le=569)):
    """Primeras filas del conjunto de datos completo."""
    return {
        "total": int(len(df_datos)),
        "registros": a_registros(df_datos, objetivo, limite),
    }
