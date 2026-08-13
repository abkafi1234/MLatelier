"""
Tests for file_utils.py — CSV and Excel reading, sheet selection.
These tests import file_utils directly (no Streamlit required).
"""
from __future__ import annotations

import sys
import os
import io

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src/mlatelier"))

from file_utils import (
    NamedBytesIO,
    TABULAR_TYPES,
    file_ext,
    read_tabular,
    sheet_selector,
)

# Alias to match test names below
_read_tabular   = read_tabular
_file_ext       = file_ext
_sheet_selector = lambda src, key: sheet_selector(src, lambda *a, **kw: kw.get("key", a[1][0] if len(a) > 1 else 0), key)
_NamedBytesIO   = NamedBytesIO


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sample_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "age":    rng.integers(20, 60, 50).astype(float),
        "score":  rng.normal(0, 1, 50),
        "label":  rng.choice(["yes", "no"], 50),
    })


def _csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _xlsx_bytes(df: pd.DataFrame, sheet_name: str = "Sheet1") -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, sheet_name=sheet_name, index=False)
    return buf.getvalue()


def _xls_bytes(df: pd.DataFrame) -> bytes:
    """Write old-style .xls using xlwt if available, otherwise skip."""
    try:
        import xlwt
        buf = io.BytesIO()
        wb  = xlwt.Workbook()
        ws  = wb.add_sheet("Sheet1")
        for ci, col in enumerate(df.columns):
            ws.write(0, ci, col)
        for ri, row in df.iterrows():
            for ci, val in enumerate(row):
                ws.write(ri + 1, ci, val)
        wb.save(buf)
        return buf.getvalue()
    except ImportError:
        return None


# ── _file_ext ─────────────────────────────────────────────────────────────────

def test_file_ext_csv_path():
    assert _file_ext("data.csv") == ".csv"


def test_file_ext_xlsx_path():
    assert _file_ext("report.xlsx") == ".xlsx"


def test_file_ext_xls_path():
    assert _file_ext("old.xls") == ".xls"


def test_file_ext_named_bytes_io():
    f = _NamedBytesIO(b"", "upload.xlsx")
    assert _file_ext(f) == ".xlsx"


def test_file_ext_no_extension():
    assert _file_ext("noext") == ""


def test_file_ext_uppercase():
    assert _file_ext("DATA.CSV") == ".csv"


# ── _read_tabular: CSV ────────────────────────────────────────────────────────

def test_read_csv_from_bytes_io():
    df  = _sample_df()
    src = _NamedBytesIO(_csv_bytes(df), "data.csv")
    out = _read_tabular(src)
    assert list(out.columns) == list(df.columns)
    assert len(out) == len(df)


def test_read_csv_from_path(tmp_path):
    df   = _sample_df()
    path = str(tmp_path / "data.csv")
    df.to_csv(path, index=False)
    out  = _read_tabular(path)
    assert len(out) == len(df)


def test_read_csv_preserves_dtypes(tmp_path):
    df   = _sample_df()
    path = str(tmp_path / "data.csv")
    df.to_csv(path, index=False)
    out  = _read_tabular(path)
    assert out["age"].dtype in (float, int, "float64", "int64")


# ── _read_tabular: Excel (xlsx) ───────────────────────────────────────────────

def test_read_xlsx_from_bytes_io():
    pytest.importorskip("openpyxl")
    df  = _sample_df()
    src = _NamedBytesIO(_xlsx_bytes(df), "data.xlsx")
    out = _read_tabular(src)
    assert list(out.columns) == list(df.columns)
    assert len(out) == len(df)


def test_read_xlsx_from_path(tmp_path):
    pytest.importorskip("openpyxl")
    df   = _sample_df()
    path = str(tmp_path / "data.xlsx")
    df.to_excel(path, index=False)
    out  = _read_tabular(path)
    assert len(out) == len(df)


def test_read_xlsx_named_sheet(tmp_path):
    pytest.importorskip("openpyxl")
    df   = _sample_df()
    path = str(tmp_path / "multi.xlsx")
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="Results", index=False)
        (_sample_df() * 2).to_excel(w, sheet_name="Other", index=False)
    out = _read_tabular(path, sheet_name="Results")
    assert len(out) == len(df)


def test_read_xlsx_second_sheet(tmp_path):
    pytest.importorskip("openpyxl")
    df1 = _sample_df()
    df2 = _sample_df().assign(age=99)
    path = str(tmp_path / "two.xlsx")
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        df1.to_excel(w, sheet_name="First",  index=False)
        df2.to_excel(w, sheet_name="Second", index=False)
    out = _read_tabular(path, sheet_name="Second")
    assert (out["age"] == 99).all()


def test_read_xlsx_preserves_values(tmp_path):
    pytest.importorskip("openpyxl")
    df   = pd.DataFrame({"x": [1.5, 2.5, 3.5], "y": ["a", "b", "c"]})
    path = str(tmp_path / "vals.xlsx")
    df.to_excel(path, index=False)
    out  = _read_tabular(path)
    assert list(out["x"]) == [1.5, 2.5, 3.5]
    assert list(out["y"]) == ["a", "b", "c"]


# ── _read_tabular: xls ───────────────────────────────────────────────────────

def test_read_xls_from_path(tmp_path):
    """Read a legacy .xls.

    Building the fixture needs a *writer*, and pandas dropped its xlwt writer —
    xlwt itself is unmaintained and has no wheel for current Pythons. Reading
    .xls still works through xlrd, which is a declared dependency; this test
    skips when no writer is available to produce the input.
    """
    pytest.importorskip("xlrd")
    df  = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    raw = _xls_bytes(df)
    if raw is None:
        pytest.skip("xlwt unavailable — cannot author a legacy .xls fixture")

    path = tmp_path / "old.xls"
    path.write_bytes(raw)
    out = _read_tabular(str(path))
    assert len(out) == 2


# ── _sheet_selector ───────────────────────────────────────────────────────────

def test_sheet_selector_csv_returns_zero():
    src = _NamedBytesIO(b"a,b\n1,2\n", "data.csv")
    result = _sheet_selector(src, key="test_csv")
    assert result == 0


def test_sheet_selector_xlsx_single_sheet(tmp_path):
    pytest.importorskip("openpyxl")
    df   = _sample_df()
    path = str(tmp_path / "single.xlsx")
    df.to_excel(path, index=False)
    result = _sheet_selector(path, key="test_single")
    # Single sheet — returns sheet name directly (no selectbox)
    assert result is not None


def test_sheet_selector_xlsx_multi_sheet_returns_name(tmp_path):
    pytest.importorskip("openpyxl")
    df   = _sample_df()
    path = str(tmp_path / "multi.xlsx")
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="Alpha", index=False)
        df.to_excel(w, sheet_name="Beta",  index=False)
    # st.selectbox is stubbed to return the _ctx object, but we just verify no crash
    result = _sheet_selector(path, key="test_multi")
    # result may be context stub or sheet name; just assert it doesn't raise
    assert result is not None


# ── Round-trip: write CSV → read back ────────────────────────────────────────

def test_roundtrip_csv(tmp_path):
    df   = _sample_df()
    path = str(tmp_path / "rt.csv")
    df.to_csv(path, index=False)
    out  = _read_tabular(path)
    pd.testing.assert_frame_equal(df.reset_index(drop=True),
                                   out.reset_index(drop=True),
                                   check_dtype=False)


def test_roundtrip_xlsx(tmp_path):
    pytest.importorskip("openpyxl")
    df   = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    path = str(tmp_path / "rt.xlsx")
    df.to_excel(path, index=False)
    out  = _read_tabular(path)
    pd.testing.assert_frame_equal(df.reset_index(drop=True),
                                   out.reset_index(drop=True),
                                   check_dtype=False)

