import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from modelo_emerrel import ejecutar_modelo

# Umbrales definidos en código (se mantienen)
UMBRAL_MIN = 9
UMBRAL_MAX = 17
umbral_usuario = st.slider("Seleccione el umbral EMEAC", min_value=UMBRAL_MIN, max_value=UMBRAL_MAX, value=15)

# ===================== helpers de post-proceso (no alteran el modelo) =====================
def usar_fechas_de_input(input_df, n_filas):
    # Si el Excel trae 'Fecha' válida para todas las filas, úsala; si no, se mantiene la generada por el modelo
    if "Fecha" in input_df.columns:
        f = pd.to_datetime(input_df["Fecha"], errors="coerce")
        if f.notna().sum() == n_filas:
            return f
    # fallback: devuelve None para no sobreescribir
    return None

def reiniciar_feb_oct(df_base, umbral_ajustable):
    """
    Recibe: DataFrame con columnas ['Fecha', 'EMERREL (0-1)']
    Devuelve: rango 1-feb → 1-oct con acumulados reiniciados y EMEAC (%) para Min/Max/Ajustable.
    """
    if df_base.empty:
        return df_base.copy()

    fechas = pd.to_datetime(df_base["Fecha"])
    years = fechas.dt.year.unique()
    yr = int(years[0]) if len(years) == 1 else int(sorted(years)[-1])

    inicio = pd.Timestamp(year=yr, month=2, day=1)   # 1 de febrero
    fin    = pd.Timestamp(year=yr, month=10, day=1)  # 1 de octubre

    vis = df_base.loc[(fechas >= inicio) & (fechas <= fin)].copy()
    if vis.empty:
        return vis

    # Reinicio de acumulado dentro del rango (no cambia constantes del modelo)
    vis["EMERREL acumulado (reiniciado)"] = vis["EMERREL (0-1)"].cumsum()

    # Curvas EMEAC (%) con tus umbrales fijos y el ajustable del slider
    vis["EMEAC (%) - Min (rango)"]       = np.clip((vis["EMERREL acumulado (reiniciado)"] / float(UMBRAL_MIN)) * 100, 0, 100)
    vis["EMEAC (%) - Max (rango)"]       = np.clip((vis["EMERREL acumulado (reiniciado)"] / float(UMBRAL_MAX)) * 100, 0, 100)
    vis["EMEAC (%) - Ajustable (rango)"] = np.clip((vis["EMERREL acumulado (reiniciado)"] / float(umbral_ajustable)) * 100, 0, 100)

    # Media móvil 5 días en el rango
    vis["EMERREL_MA5_rango"] = vis["EMERREL (0-1)"].rolling(5, min_periods=1).mean()
    return vis
# ==========================================================================================

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

    # === Ejecuta tu modelo EXACTO (no se tocan números internos) ===
    resultado = ejecutar_modelo(input_df, IW, bias_IW, LW, bias_out, umbral_usuario)

    # Si el Excel trae una columna Fecha válida, la usamos (sin alterar el modelo)
    fechas_excel = usar_fechas_de_input(input_df, len(resultado))
    if fechas_excel is not None:
        resultado["Fecha"] = fechas_excel

    # Clasificación textual (presentación)
    def clasificar_nivel(valor):
        if valor < 0.33:
            return "Bajo"
        elif valor < 0.66:
            return "Medio"
        else:
            return "Alto"
    resultado["Nivel EMERREL"] = resultado["EMERREL (0-1)"].apply(clasificar_nivel)

    # =================== Vista reiniciada: 1-feb → 1-oct ===================
    pred_vis = reiniciar_feb_oct(resultado[["Fecha","EMERREL (0-1)"]].copy(), umbral_ajustable=umbral_usuario)

    # ------------------ Tabla (rango feb–oct) ------------------
    st.subheader("Tabla de Resultados (rango 1-feb → 1-oct)")
    if pred_vis.empty:
        st.info("No hay datos en el rango 1-feb → 1-oct para el año detectado.")
    else:
        tabla = pred_vis[["Fecha", "EMERREL (0-1)", "EMEAC (%) - Ajustable (rango)"]].rename(
            columns={"EMEAC (%) - Ajustable (rango)": "EMEAC (%)"}
        )
        st.dataframe(tabla)

        # ------------------ EMERREL (rango) ------------------
        st.subheader("EMERREL (0-1) y MA5 en rango 1-feb → 1-oct (reiniciado)")
        fig1, ax1 = plt.subplots(figsize=(12, 4))
        ax1.bar(pred_vis["Fecha"], pred_vis["EMERREL (0-1)"], label="EMERREL (0-1)")
        ax1.plot(pred_vis["Fecha"], pred_vis["EMERREL_MA5_rango"], linewidth=2.2, label="Media móvil 5 días")
        ax1.set_ylabel("EMERREL (0-1)")
        ax1.set_title("EMERREL en rango 1-feb → 1-oct (acumulados reiniciados)")
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend(loc="upper right")
        st.pyplot(fig1)

        # ------------------ EMEAC (rango) con Min/Max/Ajustable ------------------
        st.markdown("**Umbrales definidos en el código:**")
        st.markdown("🔵 Umbral mínimo: 9 &nbsp;&nbsp;&nbsp;&nbsp; 🔴 Umbral máximo: 17")
        st.subheader("EMEAC (%) (rango 1-feb → 1-oct) con área sombreada entre Min y Max")

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(pred_vis["Fecha"], pred_vis["EMEAC (%) - Ajustable (rango)"], label=f"Ajustable ({umbral_usuario})", linewidth=2)
        ax.plot(pred_vis["Fecha"], pred_vis["EMEAC (%) - Min (rango)"], label="Min (9)", linestyle="--", linewidth=2)
        ax.plot(pred_vis["Fecha"], pred_vis["EMEAC (%) - Max (rango)"], label="Max (17)", linestyle="--", linewidth=2)

        ax.fill_between(pred_vis["Fecha"], pred_vis["EMEAC (%) - Min (rango)"], pred_vis["EMEAC (%) - Max (rango)"], alpha=0.3, label="Área entre Min y Max")

        ax.set_ylabel("EMEAC (%)")
        ax.set_ylim(0, 105)
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    # ------------------ (opcional) tabla completa del modelo ------------------
    st.subheader("Tabla completa (salida del modelo, sin reinicio)")
    st.dataframe(resultado[["Fecha", "Nivel EMERREL", "EMEAC (%)"]])
