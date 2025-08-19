# services/ui/pages/1_Search.py
import os, sys
import streamlit as st
from typing import Any, Dict

# Make absolute imports work when Streamlit runs pages directly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from services.ui.utils.api_client import search, similar

st.title("🔎 Search")
st.caption("Hybrid search over your catalog (lexical + vector).")

with st.sidebar:
    st.header("Search Settings")
    top_k = st.slider("Results (k)", min_value=5, max_value=50, value=20, step=5)

q = st.text_input("Query", placeholder="e.g. 'science fiction space opera'")

col1, col2 = st.columns([1, 1], vertical_alignment="bottom")
with col1:
    run = st.button("Search", type="primary", use_container_width=True)
with col2:
    st.write("")

def render_item(card: Dict[str, Any]):
    # Accept common fields; fall back to raw dict display
    title = card.get("title") or card.get("name") or card.get("item_id") or "Untitled"
    desc = card.get("description") or card.get("overview") or ""
    meta_left = []
    if "item_id" in card:
        meta_left.append(f"ID: `{card['item_id']}`")
    if "score" in card:
        meta_left.append(f"score: {round(float(card['score']), 4)}")
    st.subheader(title)
    if meta_left:
        st.caption(" • ".join(meta_left))
    if desc:
        st.write(desc)
    # Similar button
    item_id = card.get("item_id")
    if item_id:
        if st.button(f"See similar to {item_id}", key=f"sim-{item_id}"):
            try:
                sims = similar(item_id=item_id, k=10)
                st.markdown("---")
                st.write(f"**Similar to {item_id}:**")
                for s in sims:
                    st.write("- ", s.get("title") or s.get("item_id") or s)
            except Exception as e:
                st.error("Similar items failed.")
                st.code(str(e))

if run:
    if not q.strip():
        st.warning("Please enter a query.")
        st.stop()
    try:
        results = search(q.strip(), k=top_k)
    except Exception as e:
        st.error("Search failed.")
        st.code(str(e))
        st.stop()

    if not results:
        st.info("No results.")
    else:
        for r in results:
            with st.container():
                render_item(r)
                st.markdown("---")
else:
    st.info("Enter a query and click **Search**.")
