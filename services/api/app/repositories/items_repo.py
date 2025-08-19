# services/api/app/repositories/items_repo.py
from __future__ import annotations

import pandas as pd
from typing import Dict, Optional, List
from common import config

REQUIRED_ID_COLS = [
    "item_id", "id", "movieId", "product_id", "book_id"
]
TITLE_COLS = ["title", "name"]
DESC_COLS = ["description", "overview", "summary"]

class ItemsRepo:
    """
    Loads items.csv and exposes:
      - .df (normalized columns: item_id, title?, description?, text)
      - .get_item(item_id)
    Adds a 'text' column = title + " " + description (lowercased).
    """

    def __init__(self, path: Optional[str] = None):
        csv_path = path or config.ITEMS_CSV
        df = pd.read_csv(csv_path, dtype=str).fillna("")

        # find id col
        id_col = None
        lower_map = {c.lower(): c for c in df.columns}
        for c in REQUIRED_ID_COLS:
            if c.lower() in lower_map:
                id_col = lower_map[c.lower()]
                break
        if id_col is None:
            raise ValueError(f"items.csv must contain one of: {REQUIRED_ID_COLS}")

        # find title + desc
        title_col = next((lower_map[c] for c in TITLE_COLS if c in lower_map), None)
        desc_col = next((lower_map[c] for c in DESC_COLS if c in lower_map), None)

        # normalize
        rename = {id_col: "item_id"}
        if title_col: rename[title_col] = "title"
        if desc_col:  rename[desc_col]  = "description"

        df = df.rename(columns=rename)
        if "title" not in df.columns:
            df["title"] = ""
        if "description" not in df.columns:
            df["description"] = ""

        # build text field
        df["text"] = (df["title"].astype(str) + " " + df["description"].astype(str)).str.strip().str.lower()

        # keep a stable view
        self.df = df

        # map for fast lookup
        self._by_id: Dict[str, Dict] = {
            str(row["item_id"]): {
                "item_id": str(row["item_id"]),
                "title": str(row.get("title", "")),
                "description": str(row.get("description", "")),
            }
            for _, row in df.iterrows()
        }

    def get_item(self, item_id: str) -> Optional[Dict]:
        return self._by_id.get(str(item_id))

    def all_items(self) -> List[Dict]:
        return list(self._by_id.values())
