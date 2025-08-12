# -*- coding: utf-8 -*-
# meteobahia_api.py
from __future__ import annotations
import pandas as pd
import requests
import xml.etree.ElementTree as ET

def _get_attr_or_text(elem, attr="value"):
    if elem is None:
        return None
    v = elem.attrib.get(attr)
    return v if v is not None else (elem.text or None)

def parse_meteobahia_xml(xml_text: str) -> pd.DataFrame:
    """
    Parseo del XML de pronóstico diario (formato como el que adjuntaste).
    Extrae: fecha, tmax, tmin, precip.
    """
    root = ET.fromstring(xml_text)
    days = root.findall(".//tabular/day")
    rows = []
    for d in days:
        fecha  = _get_attr_or_text(d.find("fecha"))
        tmax   = _get_attr_or_text(d.find("tmax"))
        tmin   = _get_attr_or_text(d.find("tmin"))
        precip = _get_attr_or_text(d.find("precip"))
        if not fecha:
            continue
        rows.append({
            "Fecha":  pd.to_datetime(fecha, errors="coerce"),
            "TMAX":   pd.to_numeric(tmax, errors="coerce"),
            "TMIN":   pd.to_numeric(tmin, errors="coerce"),
            "Prec":   pd.to_numeric(precip, errors="coerce"),
        })
    df = pd.DataFrame(rows).dropna(subset=["Fecha"]).sort_values("Fecha").reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(columns=["Fecha","Julian_days","TMAX","TMIN","Prec"])
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    return df[["Fecha","Julian_days","TMAX","TMIN","Prec"]]

def fetch_meteobahia_api_xml(url: str, *, token: str | None = None, timeout: int = 20, params: dict | None = None) -> pd.DataFrame:
    """
    Descarga el XML desde 'url' y devuelve DataFrame normalizado.
    'token' opcional (Authorization: Bearer ...).
    """
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    r = requests.get(url, headers=headers, params=params or {}, timeout=timeout)
    r.raise_for_status()
    return parse_meteobahia_xml(r.text)
