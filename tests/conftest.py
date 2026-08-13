"""Shared test setup.

`reporting.py` binds `streamlit` as a module-level global, so whichever test
module imports it first decides what `st` is for the whole session. Installing
the stub here — before any test module is imported — makes that ordering
irrelevant. Previously each test file installed its own stub and the result
depended on alphabetical collection order.

The stub accepts any call and renders nothing; the render functions are checked
for "does not raise", with visual correctness reviewed by hand.
"""

from __future__ import annotations

import sys
import types


def make_streamlit_stub() -> types.ModuleType:
    """A streamlit module that swallows every call."""
    st = types.ModuleType("streamlit")

    class _FigStub:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def markdown(self, *a, **kw): pass
        def dataframe(self, *a, **kw): pass
        def metric(self, *a, **kw): pass
        def pyplot(self, *a, **kw): pass
        def image(self, *a, **kw): pass
        def caption(self, *a, **kw): pass
        def warning(self, *a, **kw): pass
        def error(self, *a, **kw): pass
        def success(self, *a, **kw): pass
        def info(self, *a, **kw): pass
        def text_area(self, *a, **kw): pass
        def download_button(self, *a, **kw): pass
        def button(self, *a, **kw): return False
        def columns(self, *a, **kw): return [self] * 10
        def expander(self, *a, **kw): return self
        def tabs(self, labels): return [self] * len(labels)

    _stub = _FigStub()

    for name in ["markdown", "dataframe", "metric", "pyplot", "image", "caption",
                 "warning", "error", "success", "info", "text_area",
                 "download_button", "container", "expander", "write",
                 "subheader", "divider", "spinner", "json", "code"]:
        setattr(st, name, lambda *a, _n=name, **kw: None)

    st.columns      = lambda *a, **kw: [_stub] * (a[0] if a and isinstance(a[0], int) else (len(a[0]) if a else 2))
    st.tabs         = lambda labels: [_stub] * len(labels)
    st.container    = lambda **kw: _stub
    st.expander     = lambda *a, **kw: _stub
    st.button       = lambda *a, **kw: False
    st.session_state = {}
    st.chat_input   = lambda *a, **kw: None
    st.chat_message = lambda *a, **kw: _stub
    return st


# Install before any test module imports `reporting`.
sys.modules["streamlit"] = make_streamlit_stub()
