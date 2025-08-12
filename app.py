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
    UMBRAL_MIN,
    UMBRAL_MAX,
)
from meteobahia_api import fetch_meteobahia_api_xml  # usa headers tipo navegador

st.set_page_config(page_title="MeteoBahía - EMERREL/EMEAC", layout="wide")

# ====================== Estado persistente ======================
DEFAULT_API_URL = "https://meteobahia.com.ar/scripts/forecast/for-bd.xml"
if "api_url" not in st.session_state:
    st.session_state["api_url"] = DEFAULT_API_URL
if "api_token" not in st.session_state:
    st.session_state["api_token"] = ""
if "reload_nonce" not in st.session_state:
    st.session_state["reload_nonce"] = 0
if "compat_headers" not in st.session_state:
    st.session_state["compat_headers"] = True
if "debug" not in st.session_state:
    st.session_state["debug"] = True  # deja debug ON por defecto

# ================= Sidebar =================
st.sidebar.header("Fuente de datos")
fuente = st.sidebar.radio(
    "Elegí cómo cargar datos",
    options=["API + Histórico", "Subir Excel"],
    index=0,
)

umbral_usuario = st.sidebar.slider(
    "Seleccione el umbral EMEAC (Ajustable)", min_value=UMBRAL_MIN, max_value=UMBRAL_MAX, value=15
)

st.sidebar.checkbox("Modo debug (ver diagnósticos)", key="debug")

# ============== Helpers =================
@st.cache_data(ttl=600)
def fetch_api_cached(url: str, token: str | None, nonce: int, use_browser_headers: bool):
    return fetch_meteobahia_api_xml(url.strip(), token=token or None, use_browser_headers=use_browser_headers)

def normalize_hist(df_hist: pd.DataFrame, api_year: int) -> pd.DataFrame:
    """Normaliza histórico: acepta Fecha o solo Julian_days. Valida 1–366."""
    df = df_hist.copy()
    mapping = {
        "fecha": "Fecha", "Fecha": "Fecha",
        "julian": "Julian_days", "Julian_days": "Julian_days",
        "tmax": "TMAX", "TMAX": "TMAX",
        "tmin": "TMIN", "TMIN": "TMIN",
        "prec": "Prec", "Prec": "Prec", "ppt": "Prec", "precip": "Prec",
    }
    df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
    for c in ["TMAX", "TMIN", "Prec", "Julian_days"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Validación Julian_days
    if "Julian_days" in df.columns:
        jd = df["Julian_days"]
        nonint = jd.notna() & (jd != np.floor(jd))
        out_range = jd.notna() & ((jd < 1) | (jd > 366))
        bad = nonint | out_range | jd.isna()
        if bad.any() and st.session_state["debug"]:
            st.warning(
                f"Histórico: descartadas {int(bad.sum())} filas por Julian_days inválidos "
                f"(no enteros: {int(nonint.sum())}, fuera 1–366: {int(out_range.sum())}, NaN: {int(jd.isna().sum())})"
            )
        df = df.loc[~bad].copy()
        if not df.empty:
            df["Julian_days"] = df["Julian_days"].astype(int)

    # Derivar Fecha si falta
    if "Fecha" not in df.columns and "Julian_days" in df.columns and not df.empty:
        base = pd.Timestamp(year=int(api_year), month=1, day=1)
        df["Fecha"] = df["Julian_days"].apply(lambda d: base + pd.Timedelta(days=int(d) - 1))

    if "Fecha" in df.columns:
        df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")

    if "Julian_days" not in df.columns and "Fecha" in df.columns:
        df["Julian_days"] = df["Fecha"].dt.dayofyear

    # Filtrar año incorrecto
    if "Fecha" in df.columns and not df.empty:
        wrong = df["Fecha"].dt.year != int(api_year)
        if wrong.any() and st.session_state["debug"]:
            st.warning(f"Histórico: descartadas {int(wrong.sum())} filas fuera del año {api_year}.")
        df = df.loc[~wrong].copy()

    req = {"Fecha", "Julian_days", "TMAX", "TMIN", "Prec"}
    faltan = req - set(df.columns)
    if faltan:
        raise ValueError(f"Histórico sin columnas requeridas: {faltan}")

    df = df.dropna(subset=["Fecha"]).sort_values("Fecha").reset_index(drop=True)
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    for c in ["TMAX", "TMIN", "Prec"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["Fecha", "Julian_days", "TMAX", "TMIN", "Prec"]]

def read_hist_upload(file) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame()
    try:
        if file.name.lower().endswith(".csv"):
            return pd.read_csv(file)
        return pd.read_excel(file)
    except Exception as e:
        st.error(f"No pude leer el histórico subido: {e}")
        return pd.DataFrame()

def read_hist_from_url(url: str) -> pd.DataFrame:
    if not url.strip():
        return pd.DataFrame()
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url.strip(), headers=headers, timeout=20)
        r.raise_for_status()
        content = r.content
        # Heurística simple por extensión
        if url.lower().endswith(".csv"):
            return pd.read_csv(io.BytesIO(content))
        # Default a XLSX
        return pd.read_excel(io.BytesIO(content))
    except Exception as e:
        st.error(f"No pude descargar el histórico desde la URL: {e}")
        return pd.DataFrame()

# ================= Flujo principal =================
st.title("MeteoBahía · EMERREL y EMEAC (rango 1-feb → 1-oct)")

input_df_raw = None
source_label = None

if fuente == "API + Histórico":
    st.sidebar.subheader("Pronóstico (API XML)")
    st.sidebar.text_input("URL XML", key="api_url", help="Endpoint de pronóstico (hoy → +12 días).")
    st.sidebar.text_input("Bearer token (opcional)", key="api_token", type="password")
    st.session_state["compat_headers"] = st.sidebar.checkbox(
        "Compatibilidad (headers de navegador)", value=st.session_state["compat_headers"]
    )

    # Control de recarga
    if st.sidebar.button("Actualizar ahora (forzar recarga)"):
        st.session_state["reload_nonce"] += 1

    # Histórico privado
    st.sidebar.subheader("Histórico (privado)")
    hist_file = st.sidebar.file_uploader("Subir histórico (CSV/XLSX)", type=["csv", "xlsx"])
    hist_url = st.sidebar.text_input("o pegar URL privada (CSV/XLSX)", value="")

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
            elif st.session_state["debug"]:
                st.success(f"API OK: {len(df_api)} filas, {df_api['Fecha'].min().date()} → {df_api['Fecha'].max().date()}")
        except Exception as e:
            st.error(f"Error API: {e}")
    else:
        st.info("Ingresá la URL de la API XML.")

    # 2) Histórico (archivo o URL)
    dfh_raw = pd.DataFrame()
    if hist_file is not None:
        dfh_raw = read_hist_upload(hist_file)
        if st.session_state["debug"] and not dfh_raw.empty:
            st.info(f"Histórico subido: {len(dfh_raw)} filas (sin normalizar)")
    elif hist_url.strip():
        dfh_raw = read_hist_from_url(hist_url)
        if st.session_state["debug"] and not dfh_raw.empty:
            st.info(f"Histórico (URL): {len(dfh_raw)} filas (sin normalizar)")

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
                m = (df_hist_all["Fecha"] >= start_hist) & (df_hist_all["Fecha"] <= end_hist)
                df_hist_trim = df_hist_all.loc[m].copy()
                if st.session_state["debug"] and not df_hist_trim.empty:
                    st.success(f"Hist recortado: {len(df_hist_trim)} filas, {df_hist_trim['Fecha'].min().date()} → {df_hist_trim['Fecha'].max().date()}")
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
        fuente_parts = ["API"]
        if not df_hist_trim.empty:
            fuente_parts.append(f"Hist ({df_hist_trim['Fecha'].min().date()} → {df_hist_trim['Fecha'].max().date()})")
        source_label = " + ".join(fuente_parts)

        if st.session_state["debug"]:
            st.info(f"Fusionado: {len(input_df_raw)} filas, {input_df_raw['Fecha'].min().date()} → {input_df_raw['Fecha'].max().date()}")

    else:
        st.warning("Sin datos de API. Cargá la URL y/o revisá el bloqueo 403 (probar compatibilidad).")

elif fuente == "Subir Excel":
    uploaded_file = st.file_uploader("Cargar archivo input.xlsx", type=["xlsx"])
    if uploaded_file is not None:
        try:
            input_df_raw = pd.read_excel(uploaded_file)
            source_label = f"Excel: {uploaded_file.name}"
            if st.session_state["debug"]:
                if "Fecha" in input_df_raw.columns:
                    try:
                        f = pd.to_datetime(input_df_raw["Fecha"], errors="coerce")
                        st.info(f"Excel: {len(input_df_raw)} filas, {f.min().date()} → {f.max().date()}")
                    except Exception:
                        st.info(f"Excel: {len(input_df_raw)} filas")
        except Exception as e:
            st.error(f"No pude leer el Excel: {e}")

# ================= Validación de entrada =================
if input_df_raw is None or input_df_raw.empty:
    st.stop()

# ================= Preparar datos p/ modelo =================
input_df = preparar_para_modelo(input_df_raw)
if input_df.empty:
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

# Saneo y diagnóstico
resultado["Fecha"] = pd.to_datetime(resultado["Fecha"], errors="coerce")
resultado = resultado.dropna(subset=["Fecha"]).sort_values("Fecha").reset_index(drop=True)
if not pred_vis.empty:
    pred_vis["Fecha"] = pd.to_datetime(pred_vis["Fecha"], errors="coerce")
    pred_vis = pred_vis.dropna(subset=["Fecha"]).sort_values("Fecha").reset_index(drop=True)

st.caption(f"Fuente de datos: {source_label}")
st.caption(f"Filas en rango (1-feb → 1-oct): {len(pred_vis)} · Filas totales: {len(resultado)}")

# ================= Visualización =================
if not pred_vis.empty:
    st.subheader("Tabla de Resultados (rango 1-feb → 1-oct)")
    tabla = pred_vis[["Fecha", "EMERREL (0-1)", "EMEAC (%) - Ajustable (rango)"]].rename(
        columns={"EMEAC (%) - Ajustable (rango)": "EMEAC (%)"}
    )
    st.dataframe(tabla, use_container_width=True)

    st.subheader("EMERREL (0-1) y MA5 en rango 1-feb → 1-oct (reiniciado)")
    def clasif(v): return "Bajo" if v < 0.33 else ("Medio" if v < 0.66 else "Alto")
    pred_vis["Nivel EMERREL (rango)"] = pred_vis["EMERREL (0-1)"].apply(clasif)
    color_map = {"Bajo": "green", "Medio": "yellow", "Alto": "red"}
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.bar(pred_vis["Fecha"], pred_vis["EMERREL (0-1)"],
            color=pred_vis["Nivel EMERREL (rango)"].map(color_map))
    line_ma5 = ax1.plot(pred_vis["Fecha"], pred_vis["EMERREL_MA5_rango"], linewidth=2.2, label="Media móvil 5 días")[0]
    ax1.set_ylabel("EMERREL (0-1)")
    ax1.set_title("EMERREL en rango 1-feb → 1-oct (acumulados reiniciados)")
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(handles=[Patch(facecolor=color_map[k], label=k) for k in ["Bajo","Medio","Alto"]] + [line_ma5],
               loc="upper right")
    st.pyplot(fig1); plt.close(fig1)

    st.markdown("**Umbrales definidos en el código:**")
    st.markdown(f"🔵 Min: {UMBRAL_MIN} &nbsp;&nbsp; 🔴 Max: {UMBRAL_MAX}")
    st.subheader("EMEAC (%) (rango 1-feb → 1-oct)")

    x = pred_vis["Fecha"]
    y_adj = pred_vis["EMEAC (%) - Ajustable (rango)"].astype(float).to_numpy()
    y_min = pred_vis["EMEAC (%) - Min (rango)"].astype(float).to_numpy()
    y_max = pred_vis["EMEAC (%) - Max (rango)"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, y_adj, label=f"Ajustable ({umbral_usuario})", linewidth=2)
    ax.plot(x, y_min, label=f"Min ({UMBRAL_MIN})", linestyle="--", linewidth=2)
    ax.plot(x, y_max, label=f"Max ({UMBRAL_MAX})", linestyle="--", linewidth=2)
    ax.fill_between(x, y_min, y_max, alpha=0.3, label="Área entre Min y Max")
    ax.set_ylabel("EMEAC (%)")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True)
    st.pyplot(fig); plt.close(fig)

else:
    st.warning("No hay datos en 1-feb → 1-oct. Se muestra la serie completa.")
    st.subheader("Tabla completa (salida del modelo)")
    st.dataframe(resultado[["Fecha", "EMERREL (0-1)", "EMEAC (%)"]], use_container_width=True)

    st.subheader("EMERREL (0-1) y MA5 (serie completa)")
    resultado["EMERREL_MA5"] = resultado["EMERREL (0-1)"].rolling(5, min_periods=1).mean()
    def clasif2(v): return "Bajo" if v < 0.33 else ("Medio" if v < 0.66 else "Alto")
    color_map = {"Bajo": "green", "Medio": "yellow", "Alto": "red"}
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.bar(resultado["Fecha"], resultado["EMERREL (0-1)"],
            color=resultado["EMERREL (0-1)"].apply(clasif2).map(color_map))
    line_ma5 = ax1.plot(resultado["Fecha"], resultado["EMERREL_MA5"], linewidth=2.2, label="Media móvil 5 días")[0]
    ax1.set_ylabel("EMERREL (0-1)")
    ax1.set_title("EMERREL (serie completa)")
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(handles=[Patch(facecolor=color_map[k], label=k) for k in ["Bajo","Medio","Alto"]] + [line_ma5],
               loc="upper right")
    st.pyplot(fig1); plt.close(fig1)

    st.subheader("EMEAC (%) (serie completa)")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(resultado["Fecha"], resultado["EMEAC (%)"], label="Ajustable", linewidth=2)
    ax.set_ylabel("EMEAC (%)")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True)
    st.pyplot(fig); plt.close(fig)
