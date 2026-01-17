## Overview- What makes this different?

This is a **Pure LLM-RAG system** using:
- **Gemini Embeddings API** (no local sentence-transformers needed)
- **Qdrant Vector Database** (production-ready)
- **Multi-stage LLM pipeline**
- **No issues in deployment**

Backend URL- https://rag-assessment-recommender.onrender.com
Streamlit URL- https://rag-assessment-recommenders.streamlit.app

Note:- After three weeks of inactivity, the Qdrant server became inactive.

---

## -> System Architecture

```
User Query
    ↓
Query Enhancement (Gemini) → Extract skills, level, test types
    ↓
Generate Query Embedding (Gemini API) → 768-dim vector
    ↓
Semantic Search (Qdrant) → Top 50 candidates
    ↓
Smart Balancing → Mix K (technical) + P (behavioral)
    ↓
Top 5-10 Recommendations
```

---

## > Step 1: Install Dependencies

```bash
# Create virtual environment
python -m venv .venv

# Activate
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements_rag.txt
```

---

## > Step 2: Set API Key

Create a `.env` file:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
GOOGLE_GENAI_MODEL=your_gemini_model  # best- gemini-2.5-pro
EMBED_MODEL=your_embed_model   # best- gemini-embedding-001

QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api
QDRANT_COLLECTION=collection_name
```

---

## > Step 3: Scrape SHL Catalog

```bash
python scraper.py
```

**Expected output:**
- `data/shl_assessments_complete.json` (377+ assessments)
- `data/products.csv`

---

## > Step 4: Build Vector Database

```bash
python rag_qdrant.py
```

**What happens:**
1. Loads scraped assessments
2. Creates rich text for each assessment
3. **Generates embeddings using Gemini API**
4. Creates Qdrant collection
5. Uploads vectors with metadata

**Important:** This uses Gemini's embedding API, not local models. 

---

## > Step 5: Start API Server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# To Check health
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "collection": "shl_assessments_gemini",
  "points_count": 377,
  "embedding_model": "models/text-embedding-004",
  "llm_model": "gemini-2.0-flash-exp",
  "version": "2.0 - Pure RAG"
}
```

**Test single query:**
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"Java developer with collaboration skills\", \"k\": 10}"
```

---

## > Step 6: Prepare Test Data

```bash
python convert_dataset.py
```

**Expected output:**
- `data/train_set.csv` (10 queries)
- `data/test_set.csv` (9 queries)

---

## > Step 7: Evaluate Performance

In a **new terminal** (keep API running):

```bash
python evaluate.py
```

**What happens:**
1. Tests on train set (10 queries)
2. Tests on test set (9 queries)
3. Computes metrics: Recall@5, Recall@10, MRR@10, nDCG@10
4. Generates `data/submission_predictions.csv`

---

## > Final File Structure

```
shl_recommender/
├── config.py                       # Configuration
├── rag_qdrant.py                   # Vector DB with Gemini
├── app.py                          # Pure RAG API (no sentence-transformers!)
├── evaluate.py                     # Evaluation
├── streamlit_ui.py                 # Web UI
├── scraper.py                      # Scraper
├── convert_dataset.py              # Train-Test Spliter
├── requirements_rag.txt            # Dependencies
├── .env                            # API keys
└── data/
    ├── shl_assessments_complete.json
    ├── products.csv
    ├── train_set.csv
    ├── test_set.csv
    └── submission_predictions.csv
```

---

## > Quick Start (All Commands)

```bash
# 1. Setup
python -m venv .venv
.venv\Scripts\activate  # or: source .venv/bin/activate
pip install -r requirements_pure_rag.txt

# 2. Set API key
echo "GEMINI_API_KEY=your_key_here" > .env

# 3. Scrape data (if not done)
python scraper.py

# 4. Build vector DB (IMPORTANT!)
python rag_qdrant.py

# 5. Convert datasets
python convert_dataset.py

# 6. Start API (Terminal 1)
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# 7. Evaluate (Terminal 2)
python evaluate.py

# 8. Launch UI (Terminal 3, optional)
streamlit run streamlit_ui.py
```
