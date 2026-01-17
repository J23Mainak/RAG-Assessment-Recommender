from qdrant_client.models import VectorParams, Distance, PointStruct
from qdrant_client import QdrantClient
from typing import List, Any, Dict, Optional
from dotenv import load_dotenv
from pathlib import Path
from google import genai
from tqdm import tqdm
import requests
import json
import time
import re
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "models/text-embedding-004")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "shl_assessments")
PRODUCTS_JSON = Path(os.getenv("PRODUCTS_JSON", "data/shl_assessments_complete.json"))
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM"))

if not all([GEMINI_API_KEY, EMBED_MODEL, QDRANT_URL, QDRANT_API_KEY]):
    raise RuntimeError("Please set GEMINI_API_KEY, EMBED_MODEL, QDRANT_URL and QDRANT_API_KEY in .env")

genai_client = genai.Client(api_key=GEMINI_API_KEY)

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60.0)

def extract_vector_from_item(item: Any) -> List[float]:
    if item is None:
        return []

    if hasattr(item, "values"):
        v = getattr(item, "values")
        if isinstance(v, (list, tuple)):
            return list(v)
    
    if hasattr(item, "embedding"):
        v = getattr(item, "embedding")
        if isinstance(v, (list, tuple)):
            return list(v)
    # dict types
    if isinstance(item, dict):
        if "values" in item and isinstance(item["values"], (list, tuple)):
            return list(item["values"])
        if "embedding" in item and isinstance(item["embedding"], (list, tuple)):
            return list(item["embedding"])
        
        if "data" in item and isinstance(item["data"], list) and item["data"]:
            return extract_vector_from_item(item["data"][0])
        
        for val in item.values():
            if isinstance(val, (list, tuple)):
                return list(val)
    
    if isinstance(item, (list, tuple)):
        return list(item)
    
    if hasattr(item, "value"):
        v = getattr(item, "value")
        if isinstance(v, (list, tuple)):
            return list(v)
    
    return []

def normalize_response_to_vectors(resp: Any) -> List[List[float]]:

    vectors: List[List[float]] = []
    if resp is None:
        return vectors
    if hasattr(resp, "embeddings"):
        raw = getattr(resp, "embeddings")
        if isinstance(raw, (list, tuple)):
            for it in raw:
                vec = extract_vector_from_item(it)
                if vec:
                    vectors.append(vec)
            return vectors
    # dict with 'data' or list directly
    if isinstance(resp, dict) and "data" in resp and isinstance(resp["data"], list):
        for it in resp["data"]:
            vec = extract_vector_from_item(it)
            if vec:
                vectors.append(vec)
        return vectors
    if isinstance(resp, (list, tuple)):
        for it in resp:
            vec = extract_vector_from_item(it)
            if vec:
                vectors.append(vec)
        return vectors
    # fallback single item
    single = extract_vector_from_item(resp)
    if single:
        vectors.append(single)
    return vectors

def embed_texts(texts: List[str], batch_size: int = 64, sleep_s: float = 0.05) -> List[List[float]]:
    all_vectors: List[List[float]] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]
        resp = None
        
        try:
            resp = genai_client.models.embed_content(model=EMBED_MODEL, contents=batch)
        except TypeError:
            # try alternate 'input'
            try:
                resp = genai_client.models.embed_content(model=EMBED_MODEL, input=batch)
            except Exception:
                resp = None
        except Exception as e:
            # other transient errors
            print(f"Embedding batch error: {e}")
            resp = None

        # fallback to per-item if batch failed
        if resp is None:
            batch_vectors = []
            for text in batch:
                item_resp = None
                try:
                    item_resp = genai_client.models.embed_content(model=EMBED_MODEL, contents=text)
                except Exception:
                    try:
                        item_resp = genai_client.models.embed_content(model=EMBED_MODEL, input=text)
                    except Exception as e:
                        print(f"Per-item embed failed: {e}")
                        item_resp = None
                vecs = normalize_response_to_vectors(item_resp)
                if vecs:
                    batch_vectors.append(vecs[0])
                else:
                    batch_vectors.append([])
                time.sleep(sleep_s)
            all_vectors.extend(batch_vectors)
            continue

        vectors = normalize_response_to_vectors(resp)
        all_vectors.extend(vectors)
        time.sleep(sleep_s)
    return all_vectors

def build_rich_text(a: Dict) -> str:
    parts = []
    name = str(a.get("name", "") or "").strip()
    if name:
        parts.extend([name, name])  # weight title
    desc = str(a.get("description", "") or "").strip()
    if desc and desc != "Description unavailable":
        parts.append(desc[:2000])
    if a.get("test_type"):
        parts.append(f"Test type: {a.get('test_type')}")
    if a.get("job_level"):
        parts.append(f"Job level: {a.get('job_level')}")
    if a.get("duration"):
        parts.append(f"Duration: {a.get('duration')}")
    languages = a.get("languages", [])
    if isinstance(languages, list) and languages:
        parts.append(f"Languages: {', '.join(languages)}")
    if str(a.get("remote_testing", "")).lower() == "yes":
        parts.append("Remote testing supported")
    if str(a.get("adaptive/irt_support", "")).lower() == "yes":
        parts.append("Adaptive testing supported")
    return " | ".join([p for p in parts if p])

def qdrant_search_fallback(collection: str, q_vector: List[float], limit: int = 5) -> List[Dict]:
    # 1) try qdrant-client 'search'
    try:
        if hasattr(qdrant_client, "search"):
            try:
                res = qdrant_client.search(collection_name=collection, query_vector=q_vector, limit=limit, with_payload=True)
                # res is typically list of ScoredPoint or dict-like
                results = []
                for hit in res:
                    # hit may be object with 'payload' and 'id' and 'score'
                    payload = getattr(hit, "payload", None) or (hit.get("payload") if isinstance(hit, dict) else None)
                    score = getattr(hit, "score", None) or (hit.get("score") if isinstance(hit, dict) else None)
                    _id = getattr(hit, "id", None) or (hit.get("id") if isinstance(hit, dict) else None)
                    results.append({"id": _id, "score": score, "payload": payload})
                return results
            except Exception:
                pass
    except Exception:
        pass

    # 2) try older method 'search_points'
    try:
        if hasattr(qdrant_client, "search_points"):
            try:
                res = qdrant_client.search_points(collection_name=collection, query_vector=q_vector, limit=limit, with_payload=True)
                results = []
                for hit in res:
                    # search_points may return dict items
                    payload = hit.get("payload") if isinstance(hit, dict) else getattr(hit, "payload", None)
                    results.append({"id": hit.get("id") if isinstance(hit, dict) else getattr(hit, "id", None),
                                    "score": hit.get("score") if isinstance(hit, dict) else getattr(hit, "score", None),
                                    "payload": payload})
                return results
            except Exception:
                pass
    except Exception:
        pass

    # 3) Fallback to REST API
    try:
        url = QDRANT_URL.rstrip("/") + f"/collections/{collection}/points/search"
        headers = {"api-key": QDRANT_API_KEY, "Content-Type": "application/json"}
        body = {"vector": q_vector, "limit": limit, "with_payload": True}
        r = requests.post(url, headers=headers, json=body, timeout=30)
        r.raise_for_status()
        resp = r.json()
        # resp may contain 'result' or 'hits'
        hits = resp.get("result") or resp.get("hits") or resp.get("data") or []
        results = []
        for hit in hits:
            # hit can be {id, version, score, payload} or {id, payload, score}
            payload = hit.get("payload") or (hit.get("record", {}) or {}).get("payload") or {}
            results.append({"id": hit.get("id"), "score": hit.get("score"), "payload": payload})
        return results
    except Exception as e:
        print(f"Qdrant REST search failed: {e}")
        return []


def create_vector_db():
    global EMBEDDING_DIM

    print("\n--> Creating Qdrant vector DB\n")

    if not PRODUCTS_JSON.exists():
        raise FileNotFoundError(f"Products JSON not found: {PRODUCTS_JSON}")

    with open(PRODUCTS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    # build rich texts
    valid = []
    texts = []
    metas = []
    for a in data:
        if not isinstance(a, dict):
            continue
        if not a.get("name") or not a.get("url"):
            continue
        rt = build_rich_text(a)
        if rt:
            valid.append(a)
            texts.append(rt)
            metas.append(a)

    print(f"✓ Valid assessments: {len(texts)}")

    # get embeddings
    vectors = embed_texts(texts, batch_size=64, sleep_s=0.05)

    # find first non-empty vector
    first_vec = next((v for v in vectors if v and len(v) > 0), None)
    if not first_vec:
        raise RuntimeError("No embeddings returned. Check API key, model name or quota.")

    # auto-detect embedding dim if not pinned
    if EMBEDDING_DIM <= 0:
        EMBEDDING_DIM = len(first_vec)
        print(f"Auto-detected EMBEDDING_DIM = {EMBEDDING_DIM}")
    else:
        print(f"Using EMBEDDING_DIM = {EMBEDDING_DIM}")

    # normalize vectors (pad/truncate)
    norm_vectors: List[List[float]] = []
    for v in vectors:
        if not v:
            norm_vectors.append([0.0] * EMBEDDING_DIM)
        elif len(v) < EMBEDDING_DIM:
            norm_vectors.append(v + [0.0] * (EMBEDDING_DIM - len(v)))
        elif len(v) > EMBEDDING_DIM:
            norm_vectors.append(v[:EMBEDDING_DIM])
        else:
            norm_vectors.append(v)

    # truncate to match valid length
    if len(norm_vectors) != len(valid):
        n = min(len(norm_vectors), len(valid))
        print(f"Warning: mismatch vectors vs records (vectors={len(norm_vectors)} records={len(valid)}). Truncating to {n}.")
        norm_vectors = norm_vectors[:n]
        valid = valid[:n]

    # create/recreate collection
    try:
        qdrant_client.delete_collection(QDRANT_COLLECTION)
    except Exception:
        pass

    qdrant_client.create_collection(collection_name=QDRANT_COLLECTION,
                                   vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE))
    print(f"-> Created collection: {QDRANT_COLLECTION} (dim={EMBEDDING_DIM})")

    # prepare points
    points = []
    for idx, (meta, vec) in enumerate(zip(valid, norm_vectors)):
        duration = str(meta.get("duration", "") or "")
        m = re.search(r"(\d+)", duration)
        duration_minutes = int(m.group(1)) if m else 0
        languages = meta.get("languages")
        languages_str = ", ".join(languages) if isinstance(languages, list) else str(languages or "")
        payload = {
            "name": meta.get("name"),
            "url": meta.get("url"),
            "description": meta.get("description", ""),
            "duration": duration,
            "duration_minutes": duration_minutes,
            "job_level": meta.get("job_level", ""),
            "test_type": meta.get("test_type", ""),
            "languages": languages_str,
            "remote_testing": str(meta.get("remote_testing", "")).lower() == "yes",
            "adaptive_support": str(meta.get("adaptive/irt_support", "")).lower() == "yes",
        }
        points.append(PointStruct(id=idx, vector=vec, payload=payload))

    
    print(f"\nUploading {len(points)} points to Qdrant...")
    batch = 128
    for i in tqdm(range(0, len(points), batch), desc="Upload"):
        qdrant_client.upsert(collection_name=QDRANT_COLLECTION, points=points[i:i+batch])
    print("✓ Upload complete")

    
    test_q = "Java developer assessment under 60 minutes"
    qv = embed_texts([test_q], batch_size=1, sleep_s=0.02)[0]
    # Normalize qv
    if len(qv) < EMBEDDING_DIM:
        qv = qv + [0.0] * (EMBEDDING_DIM - len(qv))
    elif len(qv) > EMBEDDING_DIM:
        qv = qv[:EMBEDDING_DIM]

    results = qdrant_search_fallback(QDRANT_COLLECTION, qv, limit=5)
    if not results:
        print("No results returned by Qdrant search (method absent or failed).")
        return

    print("\nTop results for test query:")
    for i, r in enumerate(results, 1):
        pid = r.get("id")
        score = r.get("score")
        payload = r.get("payload") or {}
        name = payload.get("name") or payload.get("title") or "<no-name>"
        print(f"{i}. id={pid} score={score:.4f} name={name}")

if __name__ == "__main__":
    create_vector_db()
