import os
import json
import time
import re
import requests
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

try:
    from google import genai
except Exception as e:
    raise ImportError("google-genai package required: pip install google-genai") from e

# Load env/config
load_dotenv()

try:
    from config import *
except Exception:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_GENAI_API_KEY", ""))
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
    GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "models/text-embedding-004")
    QDRANT_URL = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "shl_assessments")
    QDRANT_USE_MEMORY = os.getenv("QDRANT_USE_MEMORY", "False").lower() in ("1", "true", "yes")
    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
    
    # Defaults
    GEMINI_RERANK_MODEL = GEMINI_MODEL
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
    QDRANT_SEARCH_LIMIT = int(os.getenv("QDRANT_SEARCH_LIMIT", "50"))
    RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "30"))
    FINAL_K_MIN = int(os.getenv("FINAL_K_MIN", "5"))
    FINAL_K_MAX = int(os.getenv("FINAL_K_MAX", "10"))
    WEIGHT_SEMANTIC = float(os.getenv("WEIGHT_SEMANTIC", "0.4"))
    WEIGHT_LLM_RERANK = float(os.getenv("WEIGHT_LLM_RERANK", "0.5"))

# App init
app = FastAPI(title="SHL Assessment Recommender (Robust)", version="2.2")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# GenAI client init
API_KEY = GEMINI_API_KEY or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_GENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY / GOOGLE_GENAI_API_KEY must be set in environment or config.py")


genai_client = None
try:
    genai_client = genai.Client(api_key=API_KEY)
except TypeError:
    try:
        genai.configure(api_key=API_KEY)
        genai_client = genai
    except Exception:
        genai_client = None

if genai_client is None:
    raise RuntimeError("Failed to initialize google-genai client. Check installed package and API key.")

# Qdrant client init
QDRANT_URL_ENV = os.getenv("QDRANT_URL", QDRANT_URL if 'QDRANT_URL' in globals() else None)
QDRANT_API_KEY_ENV = os.getenv("QDRANT_API_KEY", QDRANT_API_KEY if 'QDRANT_API_KEY' in globals() else None)

if QDRANT_URL_ENV and QDRANT_API_KEY_ENV:
    qdrant_client = QdrantClient(url=QDRANT_URL_ENV, api_key=QDRANT_API_KEY_ENV, timeout=60.0)
    qdrant_backend = "cloud"
elif not QDRANT_USE_MEMORY and ('QDRANT_HOST' in globals() and QDRANT_HOST):
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=int(os.getenv("QDRANT_PORT", QDRANT_PORT)))
    qdrant_backend = "local"
else:
    qdrant_client = QdrantClient(":memory:")
    qdrant_backend = "memory"

print(f"-> Qdrant backend: {qdrant_backend}")

# Pydantic models
class QueryRequest(BaseModel):
    text: str
    use_ai: bool = True
    k: int = 10

class AssessmentResponse(BaseModel):
    name: str
    url: str
    description: str
    duration: str
    languages: str
    job_level: str
    remote_testing: str
    adaptive_support: str
    test_type: str
    score: float
    ai_insights: str = ""

# small caches
QUERY_CACHE: Dict[str, Dict] = {}
EMBED_CACHE: Dict[str, List[float]] = {}

def extract_vector_from_item(item) -> List[float]:
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
    if hasattr(item, "value"):
        v = getattr(item, "value")
        if isinstance(v, (list, tuple)):
            return list(v)
    if isinstance(item, dict):
        if "values" in item and isinstance(item["values"], (list, tuple)):
            return list(item["values"])
        if "embedding" in item and isinstance(item["embedding"], (list, tuple)):
            return list(item["embedding"])
        if "data" in item and isinstance(item["data"], list) and item["data"]:
            return extract_vector_from_item(item["data"][0])
        for v in item.values():
            if isinstance(v, (list, tuple)):
                return list(v)
    if isinstance(item, (list, tuple)):
        return list(item)
    return []

def normalize_response_to_vectors(resp) -> List[List[float]]:
    vectors = []
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

def get_query_embedding(query: str) -> List[float]:
    key = "emb::" + query[:200]
    if key in EMBED_CACHE:
        return EMBED_CACHE[key]

    resp = None
    try:
        if hasattr(genai_client, "models") and hasattr(genai_client.models, "embed_content"):
            resp = genai_client.models.embed_content(model=GEMINI_EMBED_MODEL, contents=query)
        else:
            resp = genai.embed_content(model=GEMINI_EMBED_MODEL, content=query, task_type="retrieval_query")
    except Exception as e:
        try:
            resp = genai_client.models.embed_content(model=GEMINI_EMBED_MODEL, input=query)
        except Exception:
            print("Embedding API error:", e)
            resp = None

    vectors = normalize_response_to_vectors(resp)
    if not vectors:
        raise HTTPException(status_code=500, detail="Embedding generation failed (no vectors returned). Check API key, model name, or quotas.")
    vec = vectors[0]


    try:
        if 'EMBEDDING_DIM' in globals() and EMBEDDING_DIM and len(vec) != EMBEDDING_DIM:
            if len(vec) < EMBEDDING_DIM:
                vec = vec + [0.0] * (EMBEDDING_DIM - len(vec))
            else:
                vec = vec[:EMBEDDING_DIM]
    except Exception:
        pass

    EMBED_CACHE[key] = vec
    return vec

def try_rest_search(collection_name: str, q_vec: List[float], limit: int = 5):
    if not QDRANT_URL_ENV or not QDRANT_API_KEY_ENV:
        return None
    url = QDRANT_URL_ENV.rstrip("/") + f"/collections/{collection_name}/points/search"
    headers = {"api-key": QDRANT_API_KEY_ENV, "Content-Type": "application/json"}
    body = {"vector": q_vec, "limit": limit, "with_payload": True}
    try:
        r = requests.post(url, headers=headers, json=body, timeout=30)
        r.raise_for_status()
        data = r.json()
        hits = data.get("result") or data.get("hits") or data.get("data") or []
        results = []
        for hit in hits:
            payload = hit.get("payload") or (hit.get("record", {}) or {}).get("payload") or {}
            results.append({"id": hit.get("id"), "score": hit.get("score"), "payload": payload})
        return results
    except Exception as e:
        
        print(f"REST search for collection '{collection_name}' failed: {e}")
        return None

def semantic_search(query_embedding: List[float], limit: int = 50) -> List[Dict]:
    # 1) try qdrant-client modern search
    try:
        if hasattr(qdrant_client, "search"):
            try:
                hits = qdrant_client.search(collection_name=QDRANT_COLLECTION, query_vector=query_embedding, limit=limit, with_payload=True)
                results = []
                for hit in hits:
                    payload = getattr(hit, "payload", None) or (hit.get("payload") if isinstance(hit, dict) else None)
                    score = getattr(hit, "score", None) or (hit.get("score") if isinstance(hit, dict) else None)
                    _id = getattr(hit, "id", None) or (hit.get("id") if isinstance(hit, dict) else None)
                    results.append({"_idx": _id, "_score": float(score or 0.0), **(payload or {})})
                return results
            except Exception as e:
                print("qdrant_client.search failed:", e)
    except Exception:
        pass

    # 2) try older search_points
    try:
        if hasattr(qdrant_client, "search_points"):
            try:
                hits = qdrant_client.search_points(collection_name=QDRANT_COLLECTION, query_vector=query_embedding, limit=limit, with_payload=True)
                results = []
                for hit in hits:
                    payload = hit.get("payload") if isinstance(hit, dict) else getattr(hit, "payload", None)
                    results.append({"_idx": hit.get("id") if isinstance(hit, dict) else getattr(hit, "id", None), "_score": float(hit.get("score") if isinstance(hit, dict) else getattr(hit, "score", 0.0)), **(payload or {})})
                return results
            except Exception as e:
                print("qdrant_client.search_points failed:", e)
    except Exception:
        pass

    # 3) fallback: try configured collection and some likely alternatives
    tried_names = []
    candidate_names = [QDRANT_COLLECTION, os.getenv("ALT_QDRANT_COLLECTION"), "shl_assessments", "shl_assessments_gemini"]
    
    candidate_names = [c for i, c in enumerate(candidate_names) if c and c not in candidate_names[:i]]
    for cname in candidate_names:
        tried_names.append(cname)
        res = try_rest_search(cname, query_embedding, limit=limit)
        if res is not None:
            # map returned results into same shape as above
            mapped = []
            for r in res:
                payload = r.get("payload") or {}
                mapped.append({"_idx": r.get("id"), "_score": float(r.get("score") or 0.0), **(payload or {})})
            print(f"REST search succeeded using collection: {cname}")
            return mapped

    print(f"All search attempts failed. Tried collections: {tried_names}")
    return []

def heuristic_enhance(query: str) -> Dict[str, Any]:
    # Basic heuristic: extract capitalized tokens as skills; detect level; detect minutes
    skills = []
    tokens = re.findall(r"\b[A-Za-z\+\#\-\_]{2,20}\b", query)

    # select unique words longer than 2 and not english stop words (simple)
    stop = {"the","and","for","with","in","to","a","an","of","on","under","over"}
    for t in tokens:
        if t.lower() not in stop and not t.isdigit() and len(t) > 2:
            skills.append(t)

    seen = set(); skills = [x for x in skills if not (x in seen or seen.add(x))]
    # detect level
    lvl = "Mid"
    if re.search(r"\b(senior|sr\.|lead|architect)\b", query, flags=re.I):
        lvl = "Senior"
    elif re.search(r"\b(entry|junior|jr\.|associate)\b", query, flags=re.I):
        lvl = "Entry"

    # duration in minutes
    m = re.search(r"(\d{1,3})\s*(?:min|minute|minutes|mins)", query, flags=re.I)
    dur = int(m.group(1)) if m else None
    enhanced_query = f"Assessments for {lvl} role focusing on: {', '.join(skills[:8])}"
    return {
        "key_skills": skills[:15],
        "role_level": lvl,
        "test_types_needed": ["K-Technical", "P-Behavioral"],
        "duration_preference": dur,
        "enhanced_query": enhanced_query,
        "min_technical_tests": 3,
        "min_behavioral_tests": 2
    }

def enhance_query_with_llm(query: str) -> Dict[str, Any]:
    # cache
    if query in QUERY_CACHE:
        return QUERY_CACHE[query]

    prompt = f"""You are an expert HR assessment consultant. Analyze this job requirement and return JSON:
    INPUT: {query}
    Return JSON with keys: key_skills (list), role_level, test_types_needed (list), duration_preference (minutes or null),
    enhanced_query (2-3 sentence rewrite), min_technical_tests, min_behavioral_tests.
    Return ONLY JSON.
    """
    
    text = None
    try:
        if hasattr(genai, "GenerativeModel"):
            try:
                model = genai.GenerativeModel(GEMINI_MODEL)
                resp = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=LLM_TEMPERATURE, max_output_tokens=LLM_MAX_TOKENS))
                text = getattr(resp, "text", None) or str(resp)
            except Exception as e:
                print("GenerativeModel.generate_content failed:", e)
                text = None

        if text is None:
            try:
                if hasattr(genai_client, "responses") and hasattr(genai_client.responses, "create"):
                    resp = genai_client.responses.create(model=GEMINI_MODEL, input=prompt, max_output_tokens=LLM_MAX_TOKENS)
                    text = ""
                    if hasattr(resp, "output"):
                        outputs = getattr(resp, "output", [])
                        for o in outputs:
                            for c in (o.get("content") or []):
                                text += c.get("text", "")
                    else:
                        text = str(resp)
            except Exception as e:
                print("genai_client.responses.create not available:", e)
                text = None
    except Exception as e:
        print("LLM enhancement primary attempts failed:", e)
        text = None

    # parse text if available
    parsed = None
    if text:
        try:
            # strip triple backticks and json fences
            if "```json" in text:
                text = text.split("```json",1)[1].split("```",1)[0].strip()
            elif "```" in text:
                text = text.split("```",1)[1].split("```",1)[0].strip()
            parsed = json.loads(text)
        except Exception as e:
            print("Failed to parse LLM JSON output:", e)
            parsed = None

    if not parsed:
        parsed = heuristic_enhance(query)

    # defaults & cache
    parsed.setdefault("key_skills", parsed.get("key_skills", []))
    parsed.setdefault("role_level", parsed.get("role_level", "Mid"))
    parsed.setdefault("test_types_needed", parsed.get("test_types_needed", ["K-Technical","P-Behavioral"]))
    parsed.setdefault("duration_preference", parsed.get("duration_preference", None))
    parsed.setdefault("enhanced_query", parsed.get("enhanced_query", query))
    parsed.setdefault("min_technical_tests", parsed.get("min_technical_tests", 3))
    parsed.setdefault("min_behavioral_tests", parsed.get("min_behavioral_tests", 2))

    QUERY_CACHE[query] = parsed
    return parsed

def rerank_with_llm(intent: Dict[str, Any], candidates: List[Dict], top_k: int = 30) -> List[tuple]:
    # If no candidates, return empty
    if not candidates:
        return []

    cand_lines = []
    for i, c in enumerate(candidates[:top_k], 1):
        name = (c.get("name") or c.get("payload", {}).get("name") or "<no-name>")
        cand_lines.append(f"{i}. {name} | Type: {c.get('test_type','N/A')} | Duration: {c.get('duration','N/A')}")
    prompt = f"Score each of these {len(cand_lines)} candidates 0.0-1.0 for relevance to: {intent.get('enhanced_query')}\n\n" + "\n".join(cand_lines) + "\nReturn JSON array only."

    text = None
    try:
        if hasattr(genai, "GenerativeModel"):
            try:
                model = genai.GenerativeModel(GEMINI_RERANK_MODEL)
                resp = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.0, max_output_tokens=512))
                text = getattr(resp, "text", None) or str(resp)
            except Exception as e:
                print("GenerativeModel rerank failed:", e)
                text = None
        if text is None:
            # try responses.create if available
            if hasattr(genai_client, "responses") and hasattr(genai_client.responses, "create"):
                resp = genai_client.responses.create(model=GEMINI_RERANK_MODEL, input=prompt, max_output_tokens=512)
                text = ""
                outputs = getattr(resp, "output", []) or []
                for o in outputs:
                    for c in (o.get("content") or []):
                        text += c.get("text", "")
    except Exception as e:
        print("Rerank attempts failed:", e)
        text = None

    scores = None
    if text:
        try:
            if "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            scores = json.loads(text)
        except Exception as e:
            print("Failed to parse rerank JSON:", e)
            scores = None

    if not scores:
        scores = []
        for c in candidates[:top_k]:
            scores.append(float(c.get("_score", c.get("score", 0.0) or 0.0)))
        # Normalize
        if scores:
            maxv = max(scores)
            if maxv > 0:
                scores = [s / maxv for s in scores]
            else:
                scores = [1.0 - (i * 0.01) for i in range(len(scores))]

    # pair (id, score); candidates order corresponds to scores
    out = []
    for i, s in enumerate(scores[:len(candidates[:top_k])]):
        cid = candidates[i].get("_idx") or candidates[i].get("id")
        out.append((cid, float(s)))
    out.sort(key=lambda x: x[1], reverse=True)
    return out

def balance_by_test_type(candidates: List[Dict], intent: Dict, target_k: int = 10) -> List[Dict]:
    min_tech = intent.get("min_technical_tests", 0)
    min_beh = intent.get("min_behavioral_tests", 0)
    tech, beh, other = [], [], []
    for c in candidates:
        if (c.get("is_technical") or str(c.get("test_type","")).upper().count("K")):
            tech.append(c)
        elif (c.get("is_behavioral") or str(c.get("test_type","")).upper().count("P")):
            beh.append(c)
        else:
            other.append(c)
    selected = []
    seen = set()
    for _ in range(min_tech):
        if tech and len(selected) < target_k:
            it = tech.pop(0)
            url = it.get("url") or it.get("payload",{}).get("url")
            if url not in seen:
                selected.append(it); seen.add(url)
    for _ in range(min_beh):
        if beh and len(selected) < target_k:
            it = beh.pop(0)
            url = it.get("url") or it.get("payload",{}).get("url")
            if url not in seen:
                selected.append(it); seen.add(url)
    rem = tech + beh + other
    for it in rem:
        if len(selected) >= target_k: break
        url = it.get("url") or it.get("payload",{}).get("url")
        if url not in seen:
            selected.append(it); seen.add(url)
    
    if len(selected) < target_k:
        for it in candidates:
            if len(selected) >= target_k: break
            url = it.get("url") or it.get("payload",{}).get("url")
            if url not in seen:
                selected.append(it); seen.add(url)
    return selected[:target_k]

def normalize_score(s):
    try:
        f = float(s)
        return max(0.0, min(1.0, f))
    except:
        return 0.0

@app.get("/health")
def health():
    # return status with collection existence flagged rather than raising
    info = {"status": "ok", "qdrant_backend": qdrant_backend}
    try:
        col = qdrant_client.get_collection(QDRANT_COLLECTION)
        info["collection_exists"] = True
        info["points_count"] = getattr(col, "points_count", None)
    except Exception:
        info["collection_exists"] = False
        # optionally try alternate common name
        try:
            alt = "shl_assessments"
            col = qdrant_client.get_collection(alt)
            info["collection_exists"] = True
            info["points_count"] = getattr(col, "points_count", None)
            info["detected_collection"] = alt
        except Exception:
            info["collection_exists"] = False
    info["embedding_model"] = GEMINI_EMBED_MODEL if 'GEMINI_EMBED_MODEL' in globals() else None
    info["llm_model"] = GEMINI_MODEL if 'GEMINI_MODEL' in globals() else None
    return info

@app.post("/recommend", response_model=List[AssessmentResponse])
def recommend(request: QueryRequest):
    q = request.text.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty query")

    # If URL, try to scrape content (best-effort)
    if q.startswith("http://") or q.startswith("https://"):
        try:
            r = requests.get(q, timeout=8, headers={"User-Agent":"Mozilla/5.0"})
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(r.text, "html.parser")
            paragraphs = soup.find_all("p")
            if paragraphs:
                q = " ".join([p.get_text(" ", strip=True) for p in paragraphs[:12]])
        except Exception:
            pass

    k = max(FINAL_K_MIN, min(request.k, FINAL_K_MAX))

    # 1. enhance
    intent = enhance_query_with_llm(q)

    # 2. embed
    try:
        q_emb = get_query_embedding(intent.get("enhanced_query", q))
    except HTTPException as e:
        raise e

    # 3. semantic search
    candidates = semantic_search(q_emb, limit=QDRANT_SEARCH_LIMIT)
    if not candidates:
        return []

    # 4. rerank
    rerank_input = candidates[:RERANK_TOP_K]
    reranked = rerank_with_llm(intent, rerank_input, top_k=RERANK_TOP_K)

    # map id->semantic score
    id_to_sem = {c["_idx"]: float(c.get("_score", 0.0)) for c in candidates}

    final = []
    for cid, llm_score in reranked:
        sem = id_to_sem.get(cid, 0.0)
        combined = WEIGHT_SEMANTIC * sem + WEIGHT_LLM_RERANK * float(llm_score)
        # find payload in candidates
        payload = next((c for c in candidates if c["_idx"] == cid), None)
        if payload:
            payload_copy = dict(payload)  # payload contains metadata fields
            payload_copy["_combined_score"] = combined
            payload_copy["_llm_score"] = llm_score
            final.append(payload_copy)

    final.sort(key=lambda x: x.get("_combined_score", 0.0), reverse=True)
    balanced = balance_by_test_type(final, intent, target_k=k)

    # format response
    out = []
    for item in balanced:
        ai_insights = ""
        if request.use_ai:
            try:
                ai_insights = generate_ai_insights(item.get("description",""), item.get("name",""), item.get("test_type",""))
            except Exception:
                ai_insights = ""
        out.append(AssessmentResponse(
            name=item.get("name",""),
            url=item.get("url",""),
            description=item.get("description",""),
            duration=str(item.get("duration","")),
            languages=str(item.get("languages","")),
            job_level=str(item.get("job_level","")),
            remote_testing="✅" if item.get("remote_testing") else "❌",
            adaptive_support="✅" if item.get("adaptive_support") else "❌",
            test_type=item.get("test_type",""),
            score=normalize_score(item.get("_combined_score", 0.0)),
            ai_insights=ai_insights
        ))

    return out


def generate_ai_insights(description: str, name: str, test_type: str) -> str:
    prompt = f"As an HR expert, provide 3 concise insights for this assessment.\nName: {name}\nType: {test_type}\nDescription: {description[:300]}"
    text = None
    try:
        if hasattr(genai, "GenerativeModel"):
            model = genai.GenerativeModel(GEMINI_MODEL)
            resp = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.4, max_output_tokens=120))
            text = getattr(resp, "text", None) or str(resp)
    except Exception:
        text = None
    if not text:
        # simple heuristic
        return "1. Key skills: unknown\n2. Ideal level: unknown\n3. Use case: unknown"
    return text.strip()

# run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
