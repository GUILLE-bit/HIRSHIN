
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from modelo_emerrel import ejecutar_modelo

# Umbrales definidos en código
UMBRAL_MIN = 9
UMBRAL_MAX = 17
umbral_usuario = st.slider("Seleccione el umbral EMEAC", min_value=UMBRAL_MIN, max_value=UMBRAL_MAX, value=15)

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

    # Mostrar tabla
    st.subheader("Tabla de Resultados")
    st.dataframe(resultado[["Fecha", "Nivel EMERREL", "EMEAC (%)"]])

    # Gráfico de barras EMERREL
    st.subheader("Niveles de EMERREL (Bajo, Medio, Alto)")
    color_map = {"Bajo": "green", "Medio": "orange", "Alto": "red"}
    bar_colors = resultado["Nivel EMERREL"].map(color_map)
    # Calcular media móvil de 5 días
    resultado["EMERREL_MA5"] = resultado["EMERREL (0-1)"].rolling(window=5, min_periods=1).mean()
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.bar(resultado["Fecha"], resultado["EMERREL (0-1)"], color=bar_colors, label="EMERREL (0-1)")
    ax1.plot(resultado["Fecha"], resultado["EMERREL_MA5"], color="blue", linewidth=2.2, label="Media móvil 5 días")
    ax1.set_ylabel("EMERREL (0-1)")
    ax1.set_title("Clasificación de niveles EMERREL")
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(loc="upper right")
    st.pyplot(fig1)

    # Gráfico de EMEAC
    st.markdown("**Umbrales definidos en el código:**")
    st.markdown("🔵 Umbral mínimo: 9 &nbsp;&nbsp;&nbsp;&nbsp; 🔴 Umbral máximo: 17")
    

    # Gráfico final de EMEAC con umbrales Min, Max y Ajustable
    st.subheader("EMEAC (%) con área sombreada entre umbrales Min y Max")

    # Cálculos para distintos umbrales
    emerrel_sum = resultado["EMERREL (0-1)"].values
    fechas = resultado["Fecha"].values

    def calc_emeac(emerrel, umbral):
        return np.clip(np.cumsum(emerrel) / umbral * 100, 0, 100)

    emeac_min = calc_emeac(emerrel_sum, 9)
    emeac_max = calc_emeac(emerrel_sum, 17)
    emeac_ajustable = resultado["EMEAC (%)"].values

    # Crear gráfico
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(fechas, emeac_ajustable, label="Ajustable (15)", color="black", linewidth=2)
    ax.plot(fechas, emeac_min, label="Min (9)", color="blue", linestyle="--", linewidth=2)
    ax.plot(fechas, emeac_max, label="Max (17)", color="red", linestyle="--", linewidth=2)

    # Rellenar el área entre curvas min y max
    ax.fill_between(fechas, emeac_min, emeac_max, color="gray", alpha=0.3, label="Área entre Min y Max")

    ax.set_ylabel("EMEAC (%)")
    ax.set_title("EMEAC (%) con área sombreada entre umbrales Min y Max")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
