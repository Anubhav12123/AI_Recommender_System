import os, argparse, pandas as pd, numpy as np, json
from scipy.sparse import csr_matrix, save_npz
from common import config

def build_ui_matrix(ratings_csv: str):
    df = pd.read_csv(ratings_csv, dtype=str)
    df["rating"] = df["rating"].astype(float)
    users = sorted(df["user_id"].unique().tolist())
    items = sorted(df["item_id"].unique().tolist())
    u_index = {u:i for i,u in enumerate(users)}
    i_index = {i:i2 for i2,i in enumerate(items)}

    rows = df["user_id"].map(u_index).values
    cols = df["item_id"].map(i_index).values
    vals = df["rating"].values
    mat = csr_matrix((vals, (rows, cols)), shape=(len(users), len(items)))
    return mat, users, items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", default=config.RATINGS_CSV)
    ap.add_argument("--out", default=os.path.join(config.ARTIFACT_DIR, "models", "cf"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    mat, users, items = build_ui_matrix(args.ratings)
    save_npz(os.path.join(args.out, "user_item_matrix.npz"), mat)
    with open(os.path.join(args.out, "user_ids.json"), "w") as f:
        json.dump(users, f)
    with open(os.path.join(args.out, "item_ids.json"), "w") as f:
        json.dump(items, f)
    print(f"[trainer] CF matrix saved to {args.out}")

if __name__ == "__main__":
    main()
