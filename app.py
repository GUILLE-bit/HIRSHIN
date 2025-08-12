# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from modelo_emerrel import ejecutar_modelo  # (intacto)
from meteobahia import (                    # (intacto)
    preparar_para_modelo,
    usar_fechas_de_input,
    reiniciar_feb_oct,
    UMBRAL_MIN,
    UMBRAL_MAX,
)
from meteobahia_api import fetch_meteobahia_api_xml  # ← actualizado con headers "tipo navegador"

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
    index=1,  # por defecto API
)

umbral_usuario = st.sidebar.slider(
    "Seleccione el umbral EMEAC (Ajustable)",
    min_value=UMBRAL_MIN, max_value=UMBRAL_MAX, value=15
)

# =================== Cache de descarga API con NONCE ===================
@st.cache_data(ttl=600)
def _fetch_api_cached(url: str, token: str | None, nonce: int, use_browser_headers: bool):
    # 'nonce' solo invalida la caché
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

    # === Normalizador de histórico adjunto (acepta: con Fecha o solo Julian_days) ===
    def _normalize_hist(df_hist: pd.DataFrame, api_year: int) -> pd.DataFrame:
        df = df_hist.copy()
        mapping = {
            "fecha": "Fecha", "Fecha": "Fecha",
            "julian": "Julian_days", "Julian_days": "Julian_days",
            "tmax": "TMAX", "TMAX": "TMAX",
            "tmin": "TMIN", "TMIN": "TMIN",
            "prec": "Prec", "Prec": "Prec", "ppt": "Prec", "precip": "Prec",
        }
        df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})

        # Tipos numéricos
        for c in ["TMAX", "TMIN", "Prec", "Julian_days"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # VALIDACIÓN estricta de Julian_days (1–366, entero)
        if "Julian_days" in df.columns:
            jd = df["Julian_days"].copy()
            nonint_mask = jd.notna() and not np.issubdtype(jd.dtype, np.integer)
            # si dtype no es entero, chequeo elemento a elemento:
            if nonint_mask:
                nonint_mask = (jd != np.floor(jd))
            else:
                nonint_mask = pd.Series(False, index=jd.index)

            out_range_mask = jd.notna() & ((jd < 1) | (jd > 366))
            nan_mask = jd.isna()
            bad_mask = nonint_mask | out_range_mask | nan_mask
            bad_count = int(bad_mask.sum())
            if bad_count:
                detalles = []
                ni = int((jd != np.floor(jd)).sum()) if len(jd) else 0
                if ni: detalles.append(f"no enteros: {ni}")
                fr = int(out_range_mask.sum()); fr and detalles.append(f"fuera de 1–366: {fr}")
                na = int(nan_mask.sum());      na and detalles.append(f"vacíos/NaN: {na}")
                st.warning(f"Se descartaron {bad_count} filas del histórico por Julian_days inválidos ({', '.join(detalles)}).")
                df = df.loc[~bad_mask].copy()
                jd = jd.loc[~bad_mask]
            if not df.empty:
                df["Julian_days"] = jd.astype(int)

        # Derivar Fecha desde Julian_days si falta
        if "Fecha" not in df.columns and "Julian_days" in df.columns and not df.empty:
            base = pd.Timestamp(year=int(api_year), month=1, day=1)
            df["Fecha"] = df["Julian_days"].apply(lambda d: base + pd.Timedelta(days=int(d) - 1))

        # Parsear Fecha
        if "Fecha" in df.columns:
            df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")

        # Si falta Julian_days, calcularlo desde Fecha
        if "Julian_days" not in df.columns and "Fecha" in df.columns:
            df["Julian_days"] = df["Fecha"].dt.dayofyear

        # Filtrar fechas fuera del año del pronóstico
        if "Fecha" in df.columns and not df.empty:
            wrong_year = df["Fecha"].dt.year != int(api_year)
            if wrong_year.any():
                cnt = int(wrong_year.sum())
                st.warning(f"Se descartaron {cnt} filas del histórico con Fecha fuera del año {api_year}.")
                df = df.loc[~wrong_year].copy()

        # Validación final
        req = {"Fecha", "Julian_days", "TMAX", "TMIN", "Prec"}
        faltan = req - set(df.columns)
        if faltan:
            raise ValueError(f"Histórico sin columnas requeridas: {faltan}")

        df = df.dropna(subset=["Fecha"]).sort_values("Fecha").reset_index(drop=True)
        if not df.empty:
            df["Julian_days"] = df["Fecha"].dt.dayofyear
            for c in ["TMAX", "TMIN", "Prec"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df[["Fecha", "Julian_days", "TMAX", "TMIN", "Prec"]]

    # Widgets con key (persisten en session_state)
    st.sidebar.text_input(
        "URL completa del XML",
        key="api_url",
        help="URL del pronóstico (hoy → +12 días). Descarga automática."
    )
    st.sidebar.text_input(
        "Bearer token (opcional)",
        key="api_token",
        type="password"
    )
    st.session_state["compat_headers"] = st.sidebar.checkbox(
        "Modo compatibilidad (headers de navegador)", value=st.session_state["compat_headers"]
    )

    # Histórico local (opcional): CSV/XLSX (como el que subiste)
    hist_file = st.sidebar.file_uploader("Histórico (opcional: CSV/XLSX)", type=["csv","xlsx"])

    # Botón que incrementa el NONCE (fuerza refetch sin rerun)
    if st.sidebar.button("Actualizar ahora (forzar recarga)"):
        st.session_state["reload_nonce"] += 1

    # Descarga automática + fusión con histórico faltante (1-ene → ayer)
    api_url = st.session_state["api_url"] or ""
    token   = st.session_state["api_token"] or ""
    compat  = bool(st.session_state["compat_headers"])

    if api_url.strip():
        try:
            with st.spinner("Descargando desde API (XML)…"):
                df_api = _fetch_api_cached(api_url, token, st.session_state["reload_nonce"], compat)

            if df_api is None or df_api.empty:
                st.warning("La API respondió sin datos o con XML vacío.")
            else:
                # Año del pronóstico y tramo a completar
                min_api_date = pd.to_datetime(df_api["Fecha"].min()).normalize()
                api_year = int(min_api_date.year)
                start_hist = pd.Timestamp(year=api_year, month=1, day=1)
                end_hist   = min_api_date - pd.Timedelta(days=1)

                # Normalizar histórico adjunto (si viene) y recortarlo al tramo faltante
                df_hist_trim = pd.DataFrame(columns=["Fecha","Julian_days","TMAX","TMIN","Prec"])
                if hist_file is not None and end_hist >= start_hist:
                    try:
                        if hist_file.name.lower().endswith(".csv"):
                            dfh_raw = pd.read_csv(hist_file)
                        else:
                            dfh_raw = pd.read_excel(hist_file)
                        df_hist_user = _normalize_hist(dfh_raw, api_year=api_year)
                        m = (df_hist_user["Fecha"] >= start_hist) & (df_hist_user["Fecha"] <= end_hist)
                        df_hist_trim = df_hist_user.loc[m].copy()
                    except Exception as e:
                        st.error(f"Error leyendo histórico adjunto: {e}")

                # Fusionar (hist recortado + api) sin duplicados por fecha
                df_all = pd.concat([df_hist_trim, df_api], ignore_index=True)
                df_all["Fecha"] = pd.to_datetime(df_all["Fecha"], errors="coerce")
                df_all = df_all.dropna(subset=["Fecha"]).sort_values("Fecha").drop_duplicates(subset=["Fecha"], keep="last").reset_index(drop=True)
                df_all["Julian_days"] = df_all["Fecha"].dt.dayofyear

                input_df_raw = df_all.copy()
                src_parts = ["API (XML)"]
                if df_hist_trim.shape[0] > 0:
                    src_parts.append(f"Histórico adjunto ({df_hist_trim['Fecha'].min().date()} → {df_hist_trim['Fecha'].max().date()})")
                source_label = " + ".join(src_parts)

        except Exception as e:
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

# ================= Vista reiniciada: 1-feb → 1-oct + saneo =================
pred_vis = reiniciar_feb_oct(
    resultado[["Fecha","EMERREL (0-1)"]].copy(),
    umbral_ajustable=umbral_usuario
)

# Saneo fechas/orden
resultado["Fecha"] = pd.to_datetime(resultado["Fecha"], errors="coerce")
resultado = resultado.dropna(subset=["Fecha"]).sort_values("Fecha").reset_index(drop=True)

if not pred_vis.empty:
    pred_vis = pred_vis.copy()
    pred_vis["Fecha"] = pd.to_datetime(pred_vis["Fecha"], errors="coerce")
    pred_vis = pred_vis.dropna(subset=["Fecha"]).sort_values("Fecha").reset_index(drop=True)

st.caption(f"Filas en rango (1-feb → 1-oct): {len(pred_vis)} · Filas totales: {len(resultado)}")
st.caption(f"Fuente de datos: {source_label}")

# ================= RANGO (si hay datos) =================
if not pred_vis.empty:
    st.subheader("Tabla de Resultados (rango 1-feb → 1-oct)")
    tabla = pred_vis[["Fecha", "EMERREL (0-1)", "EMEAC (%) - Ajustable (rango)"]].rename(
        columns={"EMEAC (%) - Ajustable (rango)": "EMEAC (%)"}
    )
    st.dataframe(tabla, use_container_width=True)

    st.subheader("EMERREL (0-1) y MA5 en rango 1-feb → 1-oct (reiniciado)")
    def _clasif(v):
        if v < 0.33: return "Bajo"
        elif v < 0.66: return "Medio"
        else: return "Alto"
    pred_vis["Nivel EMERREL (rango)"] = pred_vis["EMERREL (0-1)"].apply(_clasif)
    color_map = {"Bajo": "green", "Medio": "yellow", "Alto": "red"}
    bar_colors = pred_vis["Nivel EMERREL (rango)"].map(color_map)

    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.bar(pred_vis["Fecha"], pred_vis["EMERREL (0-1)"], color=bar_colors)
    line_ma5 = ax1.plot(pred_vis["Fecha"], pred_vis["EMERREL_MA5_rango"], linewidth=2.2, label="Media móvil 5 días")[0]
    ax1.set_ylabel("EMERREL (0-1)")
    ax1.set_title("EMERREL en rango 1-feb → 1-oct (acumulados reiniciados)")
    ax1.tick_params(axis='x', rotation=45)
    nivel_handles = [Patch(facecolor=color_map[k], label=k) for k in ["Bajo","Medio","Alto"]]
    ax1.legend(handles=nivel_handles + [line_ma5], loc="upper right")
    st.pyplot(fig1); plt.close(fig1)

    st.markdown("**Umbrales definidos en el código:**")
    st.markdown(f"🔵 Umbral mínimo: {UMBRAL_MIN} &nbsp;&nbsp;&nbsp;&nbsp; 🔴 Umbral máximo: {UMBRAL_MAX}")
    st.subheader("EMEAC (%) (rango 1-feb → 1-oct) con área sombreada entre Min y Max")

    x = pd.to_datetime(pred_vis["Fecha"])
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

# ================= FALLBACK (si el rango quedó vacío) =================
else:
    st.warning("No hay datos en el rango 1-feb → 1-oct para el año detectado. Se muestra la serie completa.")
    st.subheader("Tabla completa (salida del modelo)")
    st.dataframe(resultado[["Fecha", "EMERREL (0-1)", "EMEAC (%)"]], use_container_width=True)

    st.subheader("EMERREL (0-1) y MA5 (serie completa)")
    resultado["EMERREL_MA5"] = resultado["EMERREL (0-1)"].rolling(5, min_periods=1).mean()
    def _clasif2(v):
        if v < 0.33: return "Bajo"
        elif v < 0.66: return "Medio"
        else: return "Alto"
    color_map = {"Bajo": "green", "Medio": "yellow", "Alto": "red"}
    niveles = resultado["EMERREL (0-1)"].apply(_clasif2)
    bar_colors = niveles.map(color_map)

    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.bar(resultado["Fecha"], resultado["EMERREL (0-1)"], color=bar_colors)
    line_ma5 = ax1.plot(resultado["Fecha"], resultado["EMERREL_MA5"], linewidth=2.2, label="Media móvil 5 días")[0]
    ax1.set_ylabel("EMERREL (0-1)")
    ax1.set_title("EMERREL (serie completa)")
    ax1.tick_params(axis='x', rotation=45)
    nivel_handles = [Patch(facecolor=color_map[k], label=k) for k in ["Bajo","Medio","Alto"]]
    ax1.legend(handles=nivel_handles + [line_ma5], loc="upper right")
    st.pyplot(fig1); plt.close(fig1)

    st.subheader("EMEAC (%) (serie completa)")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(resultado["Fecha"], resultado["EMEAC (%)"], label="Ajustable", linewidth=2)
    ax.set_ylabel("EMEAC (%)")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True)
    st.pyplot(fig); plt.close(fig)
