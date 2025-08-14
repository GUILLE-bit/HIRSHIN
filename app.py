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
EMEAC_MIN = 6     # Umbral mínimo por defecto (cambia aquí)
EMEAC_MAX = 8     # Umbral máximo por defecto (cambia aquí)

# Umbral AJUSTABLE por defecto (editable en CÓDIGO) y opción de forzarlo
EMEAC_AJUSTABLE_DEF = 8                # Debe estar entre EMEAC_MIN y EMEAC_MAX
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
    # 'nonce' invalida la caché
    return fetch_meteobahia_api_xml(url.strip(), token=token or None, use_browser_headers=use_browser_headers)

def normalize_hist(df_hist: pd.DataFrame, api_year: int) -> pd.DataFrame:
    """Normaliza histórico: acepta Fecha o solo Julian_days. Valida 1–365/366 y nombres variados."""
    import calendar
    df = df_hist.copy()

    # 1) limpiar y mapear encabezados (tolerante)
    df.columns = [str(c).strip() for c in df.columns]
    low2orig = {c.lower(): c for c in df.columns}
    def has(c): return c in low2orig
    def col(c): return low2orig[c]

    ren = {}
    for cands, tgt in [
        (["fecha", "date", "fechas"], "Fecha"),
        (["julian_days", "julianday", "julian", "dia_juliano"], "Julian_days"),
        (["tmax", "t_max", "t max", "tx", "tmax(°c)"], "TMAX"),
        (["tmin", "t_min", "t min", "tn", "tmin(°c)"], "TMIN"),
        (["prec", "ppt", "precip", "lluvia", "mm", "prcp"], "Prec"),
    ]:
        for c in cands:
            if has(c):
                ren[col(c)] = tgt
                break
    df = df.rename(columns=ren)

    # 2) tipos
    if "Fecha" in df.columns:
        df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    for c in ["TMAX", "TMIN", "Prec", "Julian_days"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 3) validar Julian_days
    import numpy as _np
    leap = calendar.isleap(int(api_year))
    max_j = 366 if leap else 365
    if "Julian_days" in df.columns:
        jd = df["Julian_days"]
        nonint = jd.notna() & (jd != _np.floor(jd))
        out_range = jd.notna() & ((jd < 1) | (jd > max_j))
        nan = jd.isna()
        bad = nonint | out_range | nan
        df = df.loc[~bad].copy()
        if not df.empty and "Julian_days" in df.columns:
            df["Julian_days"] = df["Julian_days"].astype(int)

    # 4) derivar Fecha si falta y hay Julian_days
    if "Fecha" not in df.columns and "Julian_days" in df.columns and not df.empty:
        base = pd.Timestamp(int(api_year), 1, 1)
        df["Fecha"] = df["Julian_days"].astype(int).apply(lambda d: base + pd.Timedelta(days=d - 1))

    # 5) si falta Julian_days pero hay Fecha
    if "Julian_days" not in df.columns and "Fecha" in df.columns:
        df["Julian_days"] = df["Fecha"].dt.dayofyear

    # 6) filtrar fuera del año API
    if "Fecha" in df.columns and not df.empty:
        df = df.loc[df["Fecha"].dt.year == int(api_year)].copy()

    # 7) validar columnas requeridas
    req = {"Fecha", "Julian_days", "TMAX", "TMIN", "Prec"}
    faltan = req - set(df.columns)
    if faltan:
        raise ValueError(f"Histórico sin columnas requeridas: {faltan}")

    # 8) limpieza final y consistencia
    df = df.dropna(subset=["Fecha"]).sort_values("Fecha").reset_index(drop=True)
    if df.empty:
        return df
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    for c in ["TMAX", "TMIN", "Prec"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["Fecha", "Julian_days", "TMAX", "TMIN", "Prec"]]

def read_hist_upload(file) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(file) if file.name.lower().endswith(".csv") else pd.read_excel(file)
    except Exception as e:
        st.error(f"No pude leer el histórico subido: {e}")
        return pd.DataFrame()

def read_hist_from_url(url: str) -> pd.DataFrame:
    if not url.strip():
        return pd.DataFrame()
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url.strip(), headers=headers, timeout=25)
        r.raise_for_status()
        buf = io.BytesIO(r.content)
        if url.lower().endswith(".csv"):
            return pd.read_csv(buf)
        return pd.read_excel(buf)
    except Exception as e:
        st.error(f"No pude descargar el histórico desde la URL: {e}")
        return pd.DataFrame()

# ================= Flujo principal =================
st.title("PREDICCION EMERGENCIA AGRICOLA HIRSHIN")

input_df_raw = None
source_label = None

if fuente == "API + Histórico":
    # Pronóstico API
    st.sidebar.subheader("Pronóstico (API XML)")
    st.sidebar.text_input("URL XML", key="api_url", help="Endpoint (hoy → +12 días).")
    st.sidebar.text_input("Bearer token (opcional)", key="api_token", type="password")
    st.session_state["compat_headers"] = st.sidebar.checkbox(
        "Compatibilidad (headers de navegador)", value=st.session_state["compat_headers"]
    )

    # Control de recarga
    if st.sidebar.button("Actualizar ahora (forzar recarga)"):
        st.session_state["reload_nonce"] += 1

    # Histórico: archivo o URL
    st.sidebar.subheader("Histórico")
    hist_file = st.sidebar.file_uploader("Subir (CSV/XLSX)", type=["csv", "xlsx"])
    hist_url = st.sidebar.text_input("o URL (CSV/XLSX)", key="hist_url")

    api_url = st.session_state["api_url"] or ""
    token = st.session_state["api_token"] or ""
    compat = bool(st.session_state["compat_headers"])

    # 1) API
    df_api = pd.DataFrame()
    if api_url.strip():
        try:
            with st.spinner("Descargando API…"):
                df_api = fetch_api_cached(api_url, token, st.session_state["reload_nonce"], compat)
            if df_api.empty:
                st.warning("API: sin filas.")
        except Exception as e:
            st.error(f"Error API: {e}")
    else:
        st.info("Ingresá la URL de la API XML.")

    # 2) Histórico (archivo o URL)
    dfh_raw = pd.DataFrame()
    if hist_file is not None:
        dfh_raw = read_hist_upload(hist_file)
    elif hist_url.strip():
        dfh_raw = read_hist_from_url(hist_url)

    # 3) Fusión
    if not df_api.empty:
        min_api_date = pd.to_datetime(df_api["Fecha"].min()).normalize()
        api_year = int(min_api_date.year)
        start_hist = pd.Timestamp(api_year, 1, 1)
        end_hist = min_api_date - pd.Timedelta(days=1)

        df_hist_trim = pd.DataFrame(columns=["Fecha","Julian_days","TMAX","TMIN","Prec"])
        if not dfh_raw.empty and end_hist >= start_hist:
            try:
                df_hist_all = normalize_hist(dfh_raw, api_year=api_year)
                if not df_hist_all.empty:
                    m = (df_hist_all["Fecha"] >= start_hist) & (df_hist_all["Fecha"] <= end_hist)
                    df_hist_trim = df_hist_all.loc[m].copy()
                    if df_hist_trim.empty:
                        st.warning(
                            f"El histórico no aporta filas entre {start_hist.date()} y {end_hist.date()}."
                        )
                else:
                    st.warning("Histórico sin filas tras normalizar.")
            except Exception as e:
                st.error(f"Error normalizando histórico: {e}")

        df_all = pd.concat([df_hist_trim, df_api], ignore_index=True)
        df_all["Fecha"] = pd.to_datetime(df_all["Fecha"], errors="coerce")
        df_all = df_all.dropna(subset=["Fecha"]).sort_values("Fecha")
        df_all = df_all.drop_duplicates(subset=["Fecha"], keep="last").reset_index(drop=True)
        df_all["Julian_days"] = df_all["Fecha"].dt.dayofyear

        if df_all.empty:
            st.error("Fusión vacía (ni histórico válido ni API).")
            st.stop()

        input_df_raw = df_all.copy()
        src = ["API"]
        if not df_hist_trim.empty:
            src.append(f"Hist ({df_hist_trim['Fecha'].min().date()} → {df_hist_trim['Fecha'].max().date()})")
        source_label = " + ".join(src)
    else:
        st.warning("Sin datos de API. Cargá la URL o activá compatibilidad.")

elif fuente == "Subir Excel":
    uploaded_file = st.file_uploader("Cargar archivo input.xlsx", type=["xlsx"])
    if uploaded_file is not None:
        try:
            input_df_raw = pd.read_excel(uploaded_file)
            source_label = f"Excel: {uploaded_file.name}"
        except Exception as e:
            st.error(f"No pude leer el Excel: {e}")

# ================= Validación de entrada =================
if input_df_raw is None or input_df_raw.empty:
    st.stop()

# ================= Preparar datos p/ modelo =================
input_df = preparar_para_modelo(input_df_raw)
if input_df is None or input_df.empty:
    st.error("Tras preparar columnas, no quedaron filas válidas (julian_days, TMAX, TMIN, Prec).")
    st.stop()

# ================= Pesos del modelo =================
try:
    IW = np.load("IW.npy")
    bias_IW = np.load("bias_IW.npy")
    LW = np.load("LW.npy")
    bias_out = np.load("bias_out.npy")
except Exception as e:
    st.error(f"No pude cargar los pesos del modelo (.npy): {e}")
    st.stop()

# ================= Ejecutar modelo (intacto) =================
resultado = ejecutar_modelo(input_df, IW, bias_IW, LW, bias_out, umbral_usuario)

# Reemplazar Fecha por la del input original si está completa
fechas_excel = usar_fechas_de_input(input_df_raw, len(resultado))
if fechas_excel is not None:
    resultado["Fecha"] = fechas_excel

# ================= Rango 1-feb → 1-oct =================
pred_vis = reiniciar_feb_oct(resultado[["Fecha", "EMERREL (0-1)"]].copy(), umbral_ajustable=umbral_usuario)

# Sello y fuente
st.caption(f"Fuente de datos: {source_label}")
st.caption(f"Última actualización: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.caption(f"Umbral EMEAC usado: {umbral_usuario}" + (" (forzado desde código)" if usar_codigo else ""))

# ================= Gráficos + Tabla (rango 1-feb → 1-oct) =================
if not pred_vis.empty:
    # --- Cálculos previos ---
    pred_vis = pred_vis.copy()
    pred_vis["EMERREL_MA5_rango"] = pred_vis["EMERREL (0-1)"].rolling(5, min_periods=1).mean()

    # Clasificación 0.2 / 0.4
    def clasif(v): return "Bajo" if v < 0.2 else ("Medio" if v < 0.4 else "Alto")
    pred_vis["Nivel de EMERREL"] = pred_vis["EMERREL (0-1)"].apply(clasif)

    # --- Gráfico 1: EMERREL (barras + MA5) ---
    color_map = {"Bajo": "green", "Medio": "yellow", "Alto": "red"}
    fig1, ax1 = plt.subplots(figsize=(12, 4))

    # Área cerrada desde 0 hasta la MA5 (celeste claro), debajo de las barras
    ax1.fill_between(
        pred_vis["Fecha"],
        0,
        pred_vis["EMERREL_MA5_rango"],
        color="skyblue",
        alpha=0.3,
        zorder=0
    )

    # Barras por nivel de EMERREL
    ax1.bar(
        pred_vis["Fecha"],
        pred_vis["EMERREL (0-1)"],
        color=pred_vis["Nivel de EMERREL"].map(color_map)
    )

    # Línea de media móvil
    line_ma5 = ax1.plot(
        pred_vis["Fecha"],
        pred_vis["EMERREL_MA5_rango"],
        linewidth=2.2,
        label="Media móvil 5 días"
    )[0]

    ax1.set_ylabel("EMERREL (0-1)")
    ax1.set_title("EMERGENCIA RELATIVA DIARIA")
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(
        handles=[Patch(facecolor=color_map[k], label=k) for k in ["Bajo","Medio","Alto"]] + [line_ma5],
        loc="upper right"
    )
    ax1.grid(True)
    st.pyplot(fig1); plt.close(fig1)

    # --- Gráfico 2: EMEAC (%) con umbrales de código + ajustable ---
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

    # --- Tabla (después de ambos gráficos) ---
    pred_vis["Día juliano"] = pd.to_datetime(pred_vis["Fecha"]).dt.dayofyear
    tabla = pd.DataFrame({
        "Fecha": pred_vis["Fecha"],
        "Día juliano": pred_vis["Día juliano"].astype(int),
        "Nivel de EMERREL": pred_vis["Nivel de EMERREL"],
         nivel_icono = {"Bajo": "🟢 Bajo", "Medio": "🟠 Medio", "Alto": "🔴 Alto"}
        "EMEAC (%)": emeac_ajust
    })
    st.subheader("Tabla de Resultados (rango 1-feb → 1-oct)")
    st.dataframe(tabla, use_container_width=True)

    # Descarga CSV de la tabla del rango
    csv_rango = tabla.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Descargar tabla (rango) en CSV",
        data=csv_rango,
        file_name=f"tabla_rango_{pd.Timestamp.now().date()}.csv",
        mime="text/csv",
    )
else:
    st.warning("No hay datos en el rango 1-feb → 1-oct para el año detectado.")

