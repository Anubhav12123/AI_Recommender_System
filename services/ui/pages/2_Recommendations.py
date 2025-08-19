# services/ui/pages/2_Recommendations.py
import os, sys
import streamlit as st
from typing import Any, Dict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from services.ui.utils.api_client import user_recs

st.title("🎯 Recommendations")
st.caption("Personalized recommendations powered by collaborative filtering.")

with st.sidebar:
    st.header("Recommendation Settings")
    k = st.slider("Top-K", min_value=5, max_value=50, value=10, step=5)

user_id = st.text_input("User ID", placeholder="e.g. 123")

if st.button("Get Recommendations", type="primary", use_container_width=True):
    if not user_id.strip():
        st.warning("Please enter a user id.")
        st.stop()
    try:
        recs = user_recs(user_id=user_id.strip(), k=k)
    except Exception as e:
        st.error("Fetching recommendations failed.")
        st.code(str(e))
        st.stop()

    if not recs:
        st.info("No recommendations for this user.")
    else:
        st.subheader(f"Top {k} for user {user_id}")
        for r in recs:
            title = r.get("title") or r.get("name") or r.get("item_id") or "Untitled"
            meta = []
            if "item_id" in r:
                meta.append(f"ID: `{r['item_id']}`")
            if "score" in r:
                meta.append(f"score: {round(float(r['score']), 4)}")
            st.markdown(f"- **{title}**  " + ("• " + " • ".join(meta) if meta else ""))
else:
    st.info("Enter a user id and click **Get Recommendations**.")
