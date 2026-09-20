"""
Analyst workspace + dashboard (blueprint §13). Streamlit first — build the review
tool and metrics here; a React front is optional and Advanced-only.

Run:  streamlit run frontend/app.py
Phase 2+: talk to the FastAPI backend; render the document with bbox overlays,
the evidence ledger, the consistency matrix, and approve/reject controls.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Evident — Analyst", page_icon="🪪", layout="wide")
st.title("🪪 Evident — Analyst Workspace")
st.caption("Explainable KYC document intelligence · every decision traceable to evidence")

st.info(
    "Phase 0 scaffold. Next: connect to the FastAPI backend and render a case — "
    "document viewer with bounding-box overlays, the evidence ledger, the "
    "cross-document consistency matrix, and approve / reject / request-correction."
)

with st.sidebar:
    st.header("Review queue")
    st.write("Cases routed here when confidence is low or a hard anomaly fires.")
    st.metric("Pending review", "—")
    st.metric("Override rate", "—")
