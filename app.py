# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import io
import requests

from modelo_emerrel import ejecutar_modelo
from meteobahia import (
    preparar_para_modelo,
    usar_fechas_de_input,
    reiniciar_feb_oct,
)
from meteobahia_api import fetch_meteobahia_api_xml  # usa headers tipo navegador

st.set_page_config(page_title="PREDICCION EMERGENCIA AGRICOLA HIRSHIN", layout="wide")

# ====================== UMBRALES EMEAC (EDITABLES EN CÓDIGO) ======================
EMEAC_MIN = 5     # Umbral mínimo por defecto (cambia aquí)
EMEAC_MAX = 8     # Umbral máximo por defecto (cambia aquí)

# Umbral AJUSTABLE por defecto (editable en CÓDIGO) y opción de forzarlo
EMEAC_AJUSTABLE_DEF = 7                 # Debe estar entre EMEAC_MIN y EMEAC_MAX
FORZAR_AJUSTABLE_DESDE_CODIGO = False   # True = ignora el slider y usa EMEAC_AJUSTABLE_DEF

# ====================== Estado persistente ======================
DEFAULT_API_URL  = "https://meteobahia.com.ar/scripts/forecast/for-bd.xml"
DEFAULT_HIST_URL = "https://raw.githubusercontent.com/GUILLE-bit/HIRSHIN/main/data/historico.xlsx"

if "api_url" not in st.session_state:
    st.session_state["api_url"] = DEFAULT_API_URL
if "api_token" not in st.session_state:
    st.session_state["api_token"] = ""
if "hist_url" not in st.session_state:
    st.session_state["hist_url"] = DEFAULT_HIST_URL
if "reload_nonce" not in st.session_state:
    st.session_state["reload_nonce"] = 0
if "compat_headers" not in st.session_state:
    st.session_state["compat_headers"] = True

# ================= Sidebar =================
st.sidebar.header("Fuente de datos")
fuente = st.sidebar.radio(
    "Elegí cómo cargar datos",
    options=["API + Histórico", "Subir Excel"],
    index=0,
)

# Umbral ajustable: UI y/o código
usar_codigo = st.sidebar.checkbox(
    "Usar umbral ajustable desde CÓDIGO",
    value=FORZAR_AJUSTABLE_DESDE_CODIGO
)

umbral_slider = st.sidebar.slider(
    "Seleccione el umbral EMEAC (Ajustable)",
    min_value=int(EMEAC_MIN),
    max_value=int(EMEAC_MAX),
    value=int(np.clip(EMEAC_AJUSTABLE_DEF, EMEAC_MIN, EMEAC_MAX))  # arranca en el valor de código
)

# Umbral efectivo que usa la app
umbral_usuario = int(np.clip(
    EMEAC_AJUSTABLE_DEF if usar_codigo else umbral_slider,
    EMEAC_MIN, EMEAC_MAX
))

# ============== Helpers =================
@st.cache_data(ttl=600)
def fetch_api_cached(url: str, token: str | None, nonce: int, use_browser_headers: bool):
    return fetch_meteobahia_api_xml(url.strip(), token=token or None, use_browser_headers=use_browser_headers)

# Funciones read_hist_upload, read_hist_from_url, normalize_hist aquí sin cambios
# ...

# ================= Flujo principal =================
st.title("PREDICCION EMERGENCIA AGRICOLA HIRSHIN")

# (flujo de carga de datos API + histórico o Excel)
# ...

# ================= Ejecutar modelo =================
resultado = ejecutar_modelo(input_df, IW, bias_IW, LW, bias_out, umbral_usuario)

# ================= Rango 1-feb → 1-oct =================
pred_vis = reiniciar_feb_oct(resultado[["Fecha", "EMERREL (0-1)"]].copy(), umbral_ajustable=umbral_usuario)

if not pred_vis.empty:
    pred_vis["EMERREL_MA5_rango"] = pred_vis["EMERREL (0-1)"].rolling(5, min_periods=1).mean()
    def clasif(v): return "Bajo" if v < 0.2 else ("Medio" if v < 0.4 else "Alto")
    pred_vis["Nivel de EMERREL"] = pred_vis["EMERREL (0-1)"].apply(clasif)

    # --- Gráfico 1 ---
    color_map = {"Bajo": "green", "Medio": "yellow", "Alto": "red"}
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.bar(pred_vis["Fecha"], pred_vis["EMERREL (0-1)"],
            color=pred_vis["Nivel de EMERREL"].map(color_map))
    line_ma5 = ax1.plot(pred_vis["Fecha"], pred_vis["EMERREL_MA5_rango"], linewidth=2.2, label="Media móvil 5 días")[0]
    ax1.set_ylabel("EMERREL (0-1)")
    ax1.set_title("EMERGENCIA RELATIVA DIARIA")
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(handles=[Patch(facecolor=color_map[k], label=k) for k in ["Bajo","Medio","Alto"]] + [line_ma5],
               loc="upper right")
    st.pyplot(fig1); plt.close(fig1)

    # --- Gráfico 2 ---
    emerrel_rango = pred_vis["EMERREL (0-1)"].to_numpy()
    cumsum_rango = np.cumsum(emerrel_rango)
    emeac_ajust = np.clip(cumsum_rango / float(umbral_usuario) * 100.0, 0, 100)
    emeac_min   = np.clip(cumsum_rango / float(EMEAC_MIN)       * 100.0, 0, 100)
    emeac_max   = np.clip(cumsum_rango / float(EMEAC_MAX)       * 100.0, 0, 100)

    st.subheader("EMERGENCIA ACUMULADA DIARIA")
    st.markdown(f"**Umbrales:** Min={EMEAC_MIN} · Max={EMEAC_MAX} · Ajustable={umbral_usuario}")
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(pred_vis["Fecha"], emeac_ajust, label=f"Ajustable ({umbral_usuario})", linewidth=2)
    ax2.plot(pred_vis["Fecha"], emeac_min,   label=f"Min ({EMEAC_MIN})", linestyle="--", linewidth=2)
    ax2.plot(pred_vis["Fecha"], emeac_max,   label=f"Max ({EMEAC_MAX})", linestyle="--", linewidth=2)
    ax2.fill_between(pred_vis["Fecha"], emeac_min, emeac_max, alpha=0.3, label="Área entre Min y Max")
    ax2.set_ylabel("EMEAC (%)")
    ax2.set_ylim(0, 105)
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2); plt.close(fig2)

    # --- Tabla ---
    pred_vis["Día juliano"] = pd.to_datetime(pred_vis["Fecha"]).dt.dayofyear
    tabla = pd.DataFrame({
        "Fecha": pred_vis["Fecha"],
        "Día juliano": pred_vis["Día juliano"].astype(int),
        "Nivel de EMERREL": pred_vis["Nivel de EMERREL"],
        "EMEAC (%)": emeac_ajust
    })
    st.subheader("Tabla de Resultados (rango 1-feb → 1-oct)")
    st.dataframe(tabla, use_container_width=True)
    csv_rango = tabla.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Descargar tabla (rango) en CSV",
        data=csv_rango,
        file_name=f"tabla_rango_{pd.Timestamp.now().date()}.csv",
        mime="text/csv",
    )
else:
    st.warning("No hay datos en el rango 1-feb → 1-oct para el año detectado.")
