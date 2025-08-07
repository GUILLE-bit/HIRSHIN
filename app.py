
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from modelo_emerrel import ejecutar_modelo

# Definir umbrales en el código
UMBRAL_MIN = 10
UMBRAL_MAX = 20

# Slider para umbral ajustable
umbral_usuario = st.slider("Seleccione el umbral EMEAC", min_value=UMBRAL_MIN, max_value=UMBRAL_MAX, value=16)

uploaded_file = st.file_uploader("Carga tu archivo input.xlsx", type=["xlsx"])
if uploaded_file:
    input_df = pd.read_excel(uploaded_file)
    input_df = input_df.rename(columns={
        "Julian_days": "julian_days",
        "TMAX": "tmax",
        "TMIN": "Tmin",
        "Prec": "prec"
    })

    for col in ['julian_days', 'tmax', 'Tmin', 'prec']:
        input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
    input_df = input_df.dropna()

    IW = np.load("IW.npy")
    bias_IW = np.load("bias_IW.npy")
    LW = np.load("LW.npy")
    bias_out = np.load("bias_out.npy")

    resultado = ejecutar_modelo(input_df, IW, bias_IW, LW, bias_out, umbral_usuario)

    def clasificar_nivel(valor):
        if valor < 0.33:
            return "Bajo"
        elif valor < 0.66:
            return "Medio"
        else:
            return "Alto"

    resultado["Nivel EMERREL"] = resultado["EMERREL (0-1)"].apply(clasificar_nivel)
    st.dataframe(resultado[["Fecha", "Nivel EMERREL", "EMEAC (%)"]])
