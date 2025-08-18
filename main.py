import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import os, argparse, pandas as pd, numpy as np, re
from rank_bm25 import BM25Okapi
from common import storage, config
from services.indexer.utils.embeddings import get_encoder

ART_DIR = config.ARTIFACT_DIR
LEX_PATH = os.path.join(ART_DIR, "indices", "lexical.pkl")
EMBED_DIR = os.path.join(ART_DIR, "models", "embeddings")
EMB_NPY = os.path.join(EMBED_DIR, "item_vectors.npy")
IDMAP_PATH = os.path.join(EMBED_DIR, "id_mapping.json")

def simple_tokenize(text: str):
    return re.findall(r"[a-z0-9]+", str(text).lower())

def build_lexical(items_csv: str):
    os.makedirs(os.path.dirname(LEX_PATH), exist_ok=True)
    df = pd.read_csv(items_csv, dtype=str).fillna("")
    docs = [simple_tokenize(f"{r['title']} {r['description']}") for _, r in df.iterrows()]
    bm25 = BM25Okapi(docs)
    id_list = df["item_id"].tolist()
    storage.save_pickle({"bm25": bm25, "id_list": id_list}, LEX_PATH)
    print(f"[indexer] lexical index -> {LEX_PATH}")

def build_embeddings(items_csv: str, model_name: str):
    os.makedirs(EMBED_DIR, exist_ok=True)
    df = pd.read_csv(items_csv, dtype=str).fillna("")
    texts = [f"{r['title']} [SEP] {r['description']}" for _, r in df.iterrows()]
    enc = get_encoder(model_name)
    emb = enc.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    np.save(EMB_NPY, emb)
    id_map = {str(i): row["item_id"] for i, (_, row) in enumerate(df.iterrows())}
    storage.save_json(id_map, IDMAP_PATH)
    print(f"[indexer] embeddings -> {EMB_NPY}, id_map -> {IDMAP_PATH}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-all", action="store_true")
    ap.add_argument("--items", default=config.ITEMS_CSV)
    ap.add_argument("--model", default=config.EMBEDDING_MODEL)
    args = ap.parse_args()

    if args.build_all:
        build_lexical(args.items)
        build_embeddings(args.items, args.model)
    else:
        build_lexical(args.items)

if __name__ == "__main__":
    main()
