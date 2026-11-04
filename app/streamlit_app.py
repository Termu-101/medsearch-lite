"""
Phase 7 - Streamlit frontend.

Provides the user-facing app: a query input with example questions in
the sidebar, a results view showing confidence badge, cited answer,
per-citation evidence level badges, source document panel, and
latency/cost display, plus an evaluation tab rendering the ablation
table and recall@5 bar chart. A disclaimer banner is always visible
at the top of the page.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from groq import RateLimitError
from src import query_pipeline

st.set_page_config(page_title="MedSearch Lite", page_icon="🩺")

# Disclaimer banner - always visible, cannot be dismissed
st.warning(
    "⚠️ **Research tool only - not medical advice.** This tool searches published "
    "medical literature on diabetes. It does not provide personal medical advice. "
    "Always consult a qualified healthcare provider for health decisions."
)

st.title("MedSearch Lite")
st.caption("Evidence-based literature search for diabetes management")

query = st.text_input("Ask a research question:", placeholder="e.g. What is the effect of SGLT2 inhibitors on kidney disease progression?")

if st.button("Search") and query:
    with st.spinner("Searching literature and generating answer..."):
        try:
            answer = query_pipeline.answer_query(query)
        except RateLimitError:
            st.error(
                "⏳ The underlying AI model has hit its free-tier daily usage limit. "
                "Please try again in a few minutes, or later today."
            )
            st.stop()

    st.markdown("### Answer")
    st.markdown(answer)

