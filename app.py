
# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from modelo_emerrel import ejecutar_modelo
from meteobahia import (
    preparar_para_modelo,
    usar_fechas_de_input,
    reiniciar_feb_oct,
    UMBRAL_MIN,
    UMBRAL_MAX,
)
from meteobahia_api import fetch_meteobahia_api_xml  # Debe incluir headers "tipo navegador"

st.set_page_config(page_title="MeteoBahía - EMERREL/EMEAC", layout="wide")

# ====================== Estado inicial persistente ======================
DEFAULT_API_URL = "https://meteobahia.com.ar/scripts/forecast/for-bd.xml"
if "api_url" not in st.session_state:
    st.session_state["api_url"] = DEFAULT_API_URL
if "api_token" not in st.session_state:
    st.session_state["api_token"] = ""
if "reload_nonce" not in st.session_state:
    st.session_state["reload_nonce"] = 0
if "compat_headers" not in st.session_state:
    st.session_state["compat_headers"] = True  # usar headers de navegador por defecto

# ============== Sidebar: selección de fuente y opciones ==============
st.sidebar.header("Fuente de datos")
fuente = st.sidebar.radio(
    "Elegí cómo cargar datos",
    options=["Subir Excel", "API MeteoBahía (XML)"],
    index=0,
)

umbral_usuario = st.sidebar.slider(
    "Seleccione el umbral EMEAC (Ajustable)",
    min_value=UMBRAL_MIN, max_value=UMBRAL_MAX, value=15
)

# =================== Cache de descarga API con NONCE ===================
@st.cache_data(ttl=600)
def _fetch_api_cached(url: str, token: str | None, nonce: int, use_browser_headers: bool):
    # 'nonce' se usa solo para invalidar caché; no afecta la llamada
    return fetch_meteobahia_api_xml(url.strip(), token=token or None, use_browser_headers=use_browser_headers)

st.title("MeteoBahía · EMERREL y EMEAC (rango 1-feb → 1-oct)")

# ================= Obtener DataFrame de entrada =================
input_df_raw = None
source_label = None

if fuente == "Subir Excel":
    uploaded_file = st.file_uploader("Cargar archivo input.xlsx", type=["xlsx"])
    if uploaded_file is not None:
        try:
            input_df_raw = pd.read_excel(uploaded_file)
            source_label = f"Excel subido: {uploaded_file.name}"
        except Exception as e:
            st.error(f"No pude leer el Excel: {e}")

elif fuente == "API MeteoBahía (XML)":
    st.sidebar.subheader("Configuración API XML")

    # Widgets con key (persisten en session_state)
    st.sidebar.text_input(
        "URL completa del XML",
        key="api_url",
        help="URL que devuelve el XML (la app descarga automáticamente).",
        value=st.session_state["api_url"] or DEFAULT_API_URL,
    )
    st.sidebar.text_input(
        "Bearer token (opcional)",
        key="api_token",
        type="password",
        value=st.session_state["api_token"],
    )
    st.session_state["compat_headers"] = st.sidebar.checkbox(
        "Modo compatibilidad (headers de navegador)", value=st.session_state["compat_headers"]
    )

    # Botón que incrementa el NONCE (fuerza refetch sin rerun)
    if st.sidebar.button("Actualizar ahora (forzar recarga)"):
        st.session_state["reload_nonce"] += 1

    # Descarga automática cuando hay URL válida
    api_url = st.session_state["api_url"] or ""
    token = st.session_state["api_token"] or ""
    compat = bool(st.session_state["compat_headers"])

    if api_url.strip():
        try:
            with st.spinner("Descargando desde API (XML)…"):
                df_api = _fetch_api_cached(api_url, token, st.session_state["reload_nonce"], compat)
            if df_api is None or df_api.empty:
                st.warning("La API respondió sin datos o con XML vacío.")
            else:
                input_df_raw = df_api.copy()
                source_label = f"API (XML): {api_url}"
        except Exception as e:
            # Mostrar código HTTP si está disponible
            try:
                import requests
                if isinstance(e, requests.HTTPError) and e.response is not None:
                    st.error(f"Error API XML (HTTP {e.response.status_code}): {e}")
                else:
                    st.error(f"Error llamando a la API XML: {e}")
            except Exception:
                st.error(f"Error llamando a la API XML: {e}")
    else:
        st.info("Ingresá la URL real del XML para iniciar la descarga automática.")

# Si no hay datos aún, avisar y salir
if input_df_raw is None or input_df_raw.empty:
    st.info("Cargá datos mediante la opción seleccionada en la barra lateral para continuar.")
    st.stop()

# ================= Preparar datos para el modelo (solo nombres/tipos) =================
input_df = preparar_para_modelo(input_df_raw)
if input_df.empty:
    st.error("Tras preparar columnas y tipos, no quedaron filas válidas. Revisá valores numéricos en julian_days, TMAX, TMIN y Prec.")
    st.stop()

# ================= Cargar pesos del modelo (sin cambios) =================
try:
    IW = np.load("IW.npy")
    bias_IW = np.load("bias_IW.npy")
    LW = np.load("LW.npy")
    bias_out = np.load("bias_out.npy")
except Exception as e:
    st.error(f"No pude cargar los pesos del modelo (.npy): {e}")
    st.stop()

# ================= Ejecutar modelo EXACTO (no se tocan números internos) =================
resultado = ejecutar_modelo(input_df, IW, bias_IW, LW, bias_out, umbral_usuario)

# Usar fechas del input si están completas y válidas (opcional)
fechas_excel = usar_fechas_de_input(input_df_raw, len(resultado))
if fechas_excel is not None:
    resultado["Fecha"] = fechas_excel

# Clasificación textual (presentación, tabla completa)
def clasificar_nivel(valor):
    if valor < 0.33:
        return "Bajo"
    elif valor < 0.66:
        return "Medio"
    else:
        return "Alto"
resultado["Nivel EMERREL"] = resultado["EMERREL (0-1)"].apply(clasificar_nivel)

# ================= Vista reiniciada: 1-feb → 1-oct =================
pred_vis = reiniciar_feb_oct(resultado[["Fecha","EMERREL (0-1)"]].copy(), umbral_ajustable=umbral_usuario)

# ================= Encabezado / fuente =================
st.caption(f"Fuente de datos: {source_label}")

# ================= Tabla (rango feb–oct) =================
st.subheader("Tabla de Resultados (rango 1-feb → 1-oct)")
if pred_vis.empty:
    st.info("No hay datos en el rango 1-feb → 1-oct para el año detectado.")
else:
    tabla = pred_vis[["Fecha", "EMERREL (0-1)", "EMEAC (%) - Ajustable (rango)"]].rename(
        columns={"EMEAC (%) - Ajustable (rango)": "EMEAC (%)"}
    )
    st.dataframe(tabla, use_container_width=True)

    # ============ EMERREL (rango) con colores por nivel ============
    st.subheader("EMERREL (0-1) y MA5 en rango 1-feb → 1-oct (reiniciado)")
    pred_vis["Nivel EMERREL (rango)"] = pred_vis["EMERREL (0-1)"].apply(clasificar_nivel)
    color_map = {"Bajo": "green", "Medio": "yellow", "Alto": "red"}
    bar_colors = pred_vis["Nivel EMERREL (rango)"].map(color_map)

    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.bar(pred_vis["Fecha"], pred_vis["EMERREL (0-1)"], color=bar_colors)
    line_ma5 = ax1.plot(pred_vis["Fecha"], pred_vis["EMERREL_MA5_rango"], linewidth=2.2, label="Media móvil 5 días")[0]
    ax1.set_ylabel("EMERREL (0-1)")
    ax1.set_title("EMERREL en rango 1-feb → 1-oct (acumulados reiniciados)")
    ax1.tick_params(axis='x', rotation=45)

    # Leyenda: parches para niveles + línea MA5
    nivel_handles = [Patch(facecolor=color_map[k], label=k) for k in ["Bajo","Medio","Alto"]]
    ax1.legend(handles=nivel_handles + [line_ma5], loc="upper right")
    st.pyplot(fig1)

    # ============ EMEAC (rango) con Min/Max/Ajustable ============
    st.markdown("**Umbrales definidos en el código:**")
    st.markdown(f"🔵 Umbral mínimo: {UMBRAL_MIN} &nbsp;&nbsp;&nbsp;&nbsp; 🔴 Umbral máximo: {UMBRAL_MAX}")
    st.subheader("EMEAC (%) (rango 1-feb → 1-oct) con área sombreada entre Min y Max")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(pred_vis["Fecha"], pred_vis["EMEAC (%) - Ajustable (rango)"], label=f"Ajustable ({umbral_usuario})", linewidth=2)
    ax.plot(pred_vis["Fecha"], pred_vis["EMEAC (%) - Min (rango)"], label=f"Min ({UMBRAL_MIN})", linestyle="--", linewidth=2)
    ax.plot(pred_vis["Fecha"], pred_vis["EMEAC (%) - Max (rango)"], label=f"Max ({UMBRAL_MAX})", linestyle="--", linewidth=2)
    ax.fill_between(pred_vis["Fecha"], pred_vis["EMEAC (%) - Min (rango)"], pred_vis["EMEAC (%) - Max (rango)"], alpha=0.3, label="Área entre Min y Max")
    ax.set_ylabel("EMEAC (%)")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# ================= Tabla completa del modelo (sin reinicio) =================
st.subheader("Tabla completa (salida del modelo, sin reinicio)")
st.dataframe(resultado[["Fecha", "Nivel EMERREL", "EMEAC (%)"]], use_container_width=True)
