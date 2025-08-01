import streamlit as st
import numpy as np
import pandas as pd
from modelo_emerrel import ejecutar_modelo
import matplotlib.pyplot as plt

# Umbral ajustable por el usuario
umbral_usuario = st.slider("Seleccione el umbral EMEAC", min_value=1, max_value=50, value=16)

# Subida de archivo
uploaded_file = st.file_uploader("Carga tu archivo input.xlsx", type=["xlsx"])

if uploaded_file:
    input_df = pd.read_excel(uploaded_file)

    # Asegurar nombres de columnas correctos
    input_df = input_df.rename(columns={
        "Julian_days": "julian_days",
        "TMAX": "tmax",
        "TMIN": "Tmin",
        "Prec": "prec"
    })

    for col in ['julian_days', 'tmax', 'Tmin', 'prec']:
        input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

    input_df = input_df.dropna()

    # Cargar archivos de pesos
    IW = np.load('IW.npy')
    bias_IW = np.load('bias_IW.npy')
    LW = np.load('LW.npy')
    bias_out = np.load('bias_out.npy')

    # Ejecutar modelo
    resultado = ejecutar_modelo(input_df, IW, bias_IW, LW, bias_out, umbral_usuario)

    # Filtrar rango de fechas entre 1 de enero y 1 de octubre
    resultado = resultado[(resultado["Fecha"] >= "2025-01-01") & (resultado["Fecha"] <= "2025-10-01")]

    # Clasificar niveles de EMERREL
    def clasificar_nivel(valor):
        if valor < 0.33:
            return "Bajo"
        elif valor < 0.66:
            return "Medio"
        else:
            return "Alto"

    resultado["Nivel EMERREL"] = resultado["EMERREL (0-1)"].apply(clasificar_nivel)

    # Mostrar tabla
    st.dataframe(resultado)

    # Gráfico de EMERREL (línea)
    st.subheader("EMERREL (0-1)")
    st.line_chart(resultado.set_index("Fecha")["EMERREL (0-1)"])

    # Gráfico de EMEAC (%)
    st.subheader("EMEAC (%)")
    st.line_chart(resultado.set_index("Fecha")["EMEAC (%)"])

    # Gráfico de barras de EMERREL coloreado por nivel
    st.subheader("Niveles de EMERREL (Bajo, Medio, Alto)")

    color_map = {"Bajo": "green", "Medio": "orange", "Alto": "red"}
    bar_colors = resultado["Nivel EMERREL"].map(color_map)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(resultado["Fecha"], resultado["EMERREL (0-1)"], color=bar_colors)
    ax.set_ylabel("EMERREL (0-1)")
    ax.set_title("Niveles de EMERREL clasificados")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

else:
    st.warning("Por favor carga el archivo 'input.xlsx' con columnas: Julian_days, TMAX, TMIN, Prec.")