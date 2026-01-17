from urllib.parse import urlparse, urlunparse, unquote
from difflib import get_close_matches
from typing import Dict, List, Set
from math import log2
from config import *
import pandas as pd
import numpy as np
import requests
import time
import json
import re
import os

API_URL = f"http://localhost:{API_PORT}/recommend"
HEALTH_URL = f"http://localhost:{API_PORT}/health"

SUBMISSION_CSV = DATA_DIR / "submission_predictions.csv"
DETAILED_CSV = DATA_DIR / "evaluation_detailed.csv"

def normalize_url(url: str) -> str:
    if not url or not isinstance(url, str):
        return ""
    url = url.strip()
    try:
        if not url.startswith("http"):
            url = "https://" + url.lstrip("/")
        p = urlparse(url)
        scheme = p.scheme.lower()
        netloc = p.netloc.lower().replace("www.", "")
        path = unquote(p.path or "").rstrip("/")
        path = re.sub(r"/+", "/", path)
        return urlunparse((scheme, netloc, path, "", "", ""))
    except Exception:
        return url.lower().rstrip("/")

def build_url_mappings(products_file) -> Dict:
    if products_file.suffix == ".json":
        data = json.loads(products_file.read_text(encoding="utf-8"))
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(products_file, dtype=str).fillna("")

    url_col = None
    for c in ["url", "assessment_url", "link"]:
        if c in df.columns:
            url_col = c
            break
    if not url_col:
        raise RuntimeError("No URL column found in products file")

    url_map = {}
    for u in df[url_col].dropna().unique():
        n = normalize_url(u)
        if n:
            url_map[n] = u

    return {
        "url_to_canonical": url_map,
        "all_canonical": set(url_map.values())
    }

def match_url(pred_url: str, mappings: Dict) -> str:
    if not pred_url:
        return ""
    n = normalize_url(pred_url)
    if n in mappings["url_to_canonical"]:
        return mappings["url_to_canonical"][n]

    close = get_close_matches(n, mappings["url_to_canonical"].keys(), n=1, cutoff=0.75)
    if close:
        return mappings["url_to_canonical"][close[0]]
    return ""

def call_api(query: str, k: int = 10) -> List[str]:
    try:
        r = requests.post(
            API_URL,
            json={"text": query, "k": k, "use_ai": False},
            timeout=60
        )
        r.raise_for_status()
        data = r.json()
        return [x["url"] for x in data if isinstance(x, dict) and x.get("url")]
    except Exception as e:
        print(f"API error: {e}")
        return []

def recall_at_k(pred: List[str], gt: Set[str], k: int) -> float:
    if not gt:
        return 0.0
    return len(set(pred[:k]) & gt) / len(gt)

def mrr_at_k(pred: List[str], gt: Set[str], k: int) -> float:
    for i, p in enumerate(pred[:k], start=1):
        if p in gt:
            return 1.0 / i
    return 0.0

def ndcg_at_k(pred: List[str], gt: Set[str], k: int) -> float:
    dcg = 0.0
    for i, p in enumerate(pred[:k], start=1):
        if p in gt:
            dcg += 1.0 / log2(i + 1)

    ideal_hits = min(len(gt), k)
    idcg = sum(1.0 / log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_dataset(dataset_path, mappings):
    print(f"Evaluating: {dataset_path}")

    df = pd.read_csv(dataset_path, dtype=str).fillna("")
    if "Query" not in df.columns:
        df.rename(columns={df.columns[0]: "Query"}, inplace=True)

    gt_col = None
    for c in df.columns:
        if c.lower().replace(" ", "") == "assessment_url":
            gt_col = c
            break

    grouped = (
        df.groupby("Query")[gt_col].apply(list).to_dict()
        if gt_col else {q: [] for q in df["Query"].unique()}
    )

    all_r5, all_r10, all_mrr, all_ndcg = [], [], [], []
    submission_rows, detailed_rows = [], []

    for idx, (query, gt_urls) in enumerate(grouped.items(), 1):
        print(f"[{idx}/{len(grouped)}] {query[:80]}")

        preds_raw = call_api(query, k=10)
        preds = []
        seen = set()

        for p in preds_raw:
            mapped = match_url(p, mappings)
            if mapped and mapped not in seen:
                preds.append(mapped)
                seen.add(mapped)

        gt_set = {
            match_url(u, mappings)
            for u in gt_urls
            if match_url(u, mappings)
        }

        r5 = recall_at_k(preds, gt_set, 5)
        r10 = recall_at_k(preds, gt_set, 10)
        mrr = mrr_at_k(preds, gt_set, 10)
        ndcg = ndcg_at_k(preds, gt_set, 10)

        if gt_set:
            all_r5.append(r5)
            all_r10.append(r10)
            all_mrr.append(mrr)
            all_ndcg.append(ndcg)

        print(f"   R@5={r5:.3f} R@10={r10:.3f} MRR={mrr:.3f} nDCG={ndcg:.3f}\n")

        for u in preds[:10]:
            submission_rows.append({"Query": query, "Assessment_url": u})

        detailed_rows.append({
            "Query": query,
            "predicted_urls": json.dumps(preds),
            "ground_truth": json.dumps(list(gt_set)),
            "recall@5": r5,
            "recall@10": r10,
            "mrr@10": mrr,
            "ndcg@10": ndcg
        })

        time.sleep(0.05)

    print("FINAL METRICS:-")
    print(f"Mean Recall@5 : {np.mean(all_r5):.4f}")
    print(f"Mean Recall@10: {np.mean(all_r10):.4f}")
    print(f"Mean MRR@10   : {np.mean(all_mrr):.4f}")
    print(f"Mean nDCG@10  : {np.mean(all_ndcg):.4f}")

    pd.DataFrame(detailed_rows).to_csv(DETAILED_CSV, index=False)
    pd.DataFrame(submission_rows).drop_duplicates().to_csv(SUBMISSION_CSV, index=False)

    print(f"Saved detailed: {DETAILED_CSV}")
    print(f"Saved submission: {SUBMISSION_CSV}")


def main():
    print("\nSHL Recommender — Evaluation\n")

    r = requests.get(HEALTH_URL, timeout=5)
    r.raise_for_status()
    print("-> API is healthy\n")

    products_file = PRODUCTS_JSON if PRODUCTS_JSON.exists() else PRODUCTS_CSV
    mappings = build_url_mappings(products_file)

    if TRAIN_SET.exists():
        evaluate_dataset(TRAIN_SET, mappings)
    if TEST_SET.exists():
        evaluate_dataset(TEST_SET, mappings)

if __name__ == "__main__":
    main()
