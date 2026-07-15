"""Thin helpers for reading tabular files (CSV + Excel) used by app.py and tests."""
from __future__ import annotations

import io
import os

import pandas as pd


TABULAR_TYPES = ["csv", "xlsx", "xls"]


class NamedBytesIO(io.BytesIO):
    """BytesIO that carries a .name attribute (mirrors Streamlit UploadedFile)."""
    def __init__(self, content: bytes, name: str):
        super().__init__(content)
        self.name = name


def file_ext(src) -> str:
    """Return lowercase extension of an uploaded file or path string."""
    name = src if isinstance(src, str) else getattr(src, "name", "")
    return os.path.splitext(name)[1].lower()


def read_tabular(src, sheet_name=0) -> pd.DataFrame:
    """
    Read a CSV or Excel file into a DataFrame.
    src may be a Streamlit UploadedFile, a NamedBytesIO, or a local path str.
    For .xlsx/.xls the sheet_name parameter is forwarded to pd.read_excel.
    Missing columns filled with 0 is the caller's responsibility.
    """
    ext = file_ext(src)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(src, sheet_name=sheet_name)
    return pd.read_csv(src)


def sheet_selector(src, selectbox_fn, key: str):
    """
    If src is an Excel file with multiple sheets, call selectbox_fn to let the
    user pick one and return the chosen name.
    Returns 0 (first sheet) for CSV or single-sheet Excel.

    selectbox_fn(label, options, key) must match Streamlit's st.selectbox signature.
    Passing it as a parameter keeps this module Streamlit-free for testing.
    """
    ext = file_ext(src)
    if ext not in (".xlsx", ".xls"):
        return 0
    try:
        xl     = pd.ExcelFile(src)
        sheets = xl.sheet_names
        if len(sheets) > 1:
            return selectbox_fn(
                f"Sheet ({len(sheets)} available)", sheets, key=key)
        return sheets[0]
    except Exception:
        return 0
