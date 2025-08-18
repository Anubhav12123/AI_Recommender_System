# AI Reco System (Hybrid Search + Recommendations)

This is a production-minded **hybrid search & recommendation** project that showcases both **ML/AI** depth (embeddings, CF)
and **systems engineering** (API service, indexer, UI separation, caching).

## Quick Start (Local)

```bash
# 1) Create venv (Python 3.10+ recommended)
python -m venv .venv && source .venv/bin/activate

# 2) Install requirements for all services
pip install -r services/api/requirements.txt
pip install -r services/indexer/requirements.txt
pip install -r services/trainer/requirements.txt
pip install -r services/ui/requirements.txt

# 3) Build artifacts (indexes + CF model)
python services/indexer/main.py --build-all
python services/trainer/train_cf.py --ratings data/raw/ratings.csv --out data/artifacts/models/cf

# 4) Run the API (in one terminal)
uvicorn services.api.app.main:app --host 0.0.0.0 --port 8000 --reload

# 5) Run the UI (in another terminal)
streamlit run services/ui/app.py --server.port 8501
```

Then open UI at http://localhost:8501.

## What’s Included
- **API (FastAPI)**: `/search`, `/items/{id}/similar`, `/users/{id}/recommendations`, `/feedback`
- **Indexer**: Builds BM25-like lexical store and FAISS vector store from item metadata
- **Trainer**: Trains item-based CF from ratings
- **UI (Streamlit)**: 4 pages: Search, Recommendations, Metrics, Admin/Indexing
- **Common**: shared schemas, ranking, metrics functions, storage utils
- **Data**: small demo CSVs under `data/raw`

## Notes
- Lexical search uses `rank-bm25` locally (lightweight). It can be replaced by OpenSearch later.
- Vector search uses `sentence-transformers` + optional FAISS. If FAISS isn't installed, a sklearn fallback is used.
- CF uses item-based kNN with cosine similarity on a sparse user-item matrix.
