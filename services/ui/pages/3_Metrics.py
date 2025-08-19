# services/ui/pages/3_Metrics.py
import os, sys
import streamlit as st

# Ensure project root on sys.path for absolute imports when Streamlit runs the page
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from services.ui.utils.api_client import get_cf_metrics

st.title("Metrics & Evaluation")
st.caption("Offline evaluation of Collaborative Filtering with leave-one-out per user.")

with st.sidebar:
    st.header("Evaluation Params")
    k = st.slider("Top-K", min_value=1, max_value=50, value=10, step=1)
    sample_users = st.number_input("Sample users (optional)", min_value=0, value=500, step=50)
    run_btn = st.button("Run Evaluation", use_container_width=True)

info = st.empty()

def run_eval():
    info.info("Running evaluation…")
    try:
        res = get_cf_metrics(k=k, sample_users=int(sample_users) if sample_users else None)
    except Exception as e:
        info.error("Evaluation failed.")
        st.code(str(e))
        st.stop()
    info.success("Done.")

    st.subheader("Results")
    st.write({
        "Users evaluated": res["users_evaluated"],
        "k": res["k"],
        "Precision@k": res["precision_at_k"],
        "Recall@k": res["recall_at_k"],
        "MAP@k": res["map_at_k"],
        "NDCG@k": res["ndcg_at_k"],
        "Hit Rate": res["hit_rate"],
        "Item coverage": res["item_coverage"],
        "Elapsed (sec)": res["elapsed_sec"],
    })

    st.bar_chart({
        "precision@k": [res["precision_at_k"]],
        "recall@k": [res["recall_at_k"]],
        "map@k": [res["map_at_k"]],
        "ndcg@k": [res["ndcg_at_k"]],
        "hit_rate": [res["hit_rate"]],
    })

if run_btn:
    run_eval()
else:
    st.info("Set parameters in the sidebar and click **Run Evaluation**.")
