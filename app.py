import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
st.set_page_config(page_title="AI Reco UI", layout="wide")
st.title("AI Reco System")
st.write("Use the sidebar to navigate pages: Search, Recommendations, Metrics, Admin.")
