# Visual Product Search

Multimodal product search — upload (or photograph) a product image, optionally add a text query, and get back visually + semantically similar catalog items plus an AI-generated recommendation, comparison, or authenticity assessment.

## Stack

| Layer | Tech |
|---|---|
| Embeddings | CLIP (`ViT-B/32`) — image + text encoded into the same 512-dim space |
| Object detection | YOLOv8 (optional auto-crop on upload) |
| Vector DB | Qdrant — dual named-vector collection (`image`, `text`) with payload filters |
| Retrieval | Reciprocal Rank Fusion (RRF) over image + text rankings, deduped by base product ID |
| LLM | Claude (`claude-sonnet-4-20250514`) via Anthropic SDK — advises on retrieved products |
| Backend | FastAPI |
| Frontend | Streamlit |
| Catalog | [`Marqo/deepfashion-inshop`](https://huggingface.co/datasets/Marqo/deepfashion-inshop) (5k items, men's apparel) |

## Architecture

```
                    ┌─────────────┐
   image + text ──▶ │  Streamlit  │
   query            │  (frontend) │
                    └──────┬──────┘
                           │ POST /search
                           ▼
                    ┌─────────────┐      ┌────────┐
                    │   FastAPI   │◀────▶│ Qdrant │
                    │  (backend)  │      └────────┘
                    │             │
                    │ CLIP encode │      ┌──────────┐
                    │ RRF fuse    │◀────▶│  Claude  │
                    │ dedupe      │      │   LLM    │
                    └─────────────┘      └──────────┘
```

**Search flow** (`backend/rag.py::search_products`):
1. Encode uploaded image (+ text query, if given) with CLIP
2. `parse_query` extracts structured constraints (color / category / gender) from free text and builds a Qdrant hard filter; falls back to unfiltered search if the filter yields nothing
3. Run image-vector and text-vector searches against Qdrant in parallel rankings
4. Fuse both rankings with **Reciprocal Rank Fusion** (`rrf_fuse`)
5. Deduplicate by base product ID (`get_base_id` strips view/angle suffixes like `_01_3_back`) — keeps the highest-scoring variant per product
6. Top-K results go into a Claude prompt (`backend/prompts.py`) tailored to the chosen intent (`recommend` / `compare` / `authentic`)

## Project structure

```
visual-search/
├── backend/
│   ├── main.py        FastAPI app — /search, /health, response models
│   ├── rag.py         Qdrant search, RRF fusion, dedup, query parsing
│   ├── vision.py      CLIP encoding + YOLO crop
│   ├── prompts.py     LLM prompt templates per intent
│   ├── config.py      Centralized constants (no magic numbers)
│   └── test_rag.py    Unit tests for pure functions
├── frontend/
│   └── app.py         Streamlit UI
├── ingest.py          Loads HF dataset, embeds, upserts into Qdrant
├── evaluate.py        Offline Recall@5 eval harness
├── docker-compose.yml Qdrant service
├── Dockerfile.backend / Dockerfile.frontend
├── Makefile           db / ingest / backend / frontend / all / stop
└── requirements.txt
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create `.env` in the project root:

```
ANTHROPIC_API_KEY=your-key-here
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

## Running it

```bash
# 1. start Qdrant (vector db, docker)
make db

# 2. ingest the catalog (one-time — embeds 5k items into Qdrant)
make ingest

# 3. start the backend (FastAPI, :8000)
make backend

# 4. start the frontend (Streamlit) — separate terminal
make frontend
```

Or `make all` to bring everything up, `make stop` to tear it down.

## Using it

1. Open the Streamlit UI, upload a product photo (or use the camera)
2. Optionally add a text query (e.g. `"blue tshirt for men"`) and pick auto-crop (YOLO)
3. Choose an intent:
   - **recommend** — picks the best match and explains why
   - **compare** — compares the top 3 across price / category / quality
   - **authentic** — flags signals suggesting genuine vs. suspicious listings
4. Get back the AI's answer plus the ranked, deduped product matches with images, prices, and similarity scores

## API

- `GET /health` — liveness check
- `POST /search` — multipart form: `file` (image), `text_query`, `intent`, `use_yolo` → returns `SearchResponse` (products, AI answer, intent, parsed query filters)
- `GET /docs` — interactive OpenAPI schema

## Evaluation

```bash
python evaluate.py
```

Runs 10 fixed text queries against `/search`, checks whether the expected category appears in the top-5, and reports **Recall@5**.

## Tests

```bash
cd backend && pytest test_rag.py -v
```

Unit tests for the pure retrieval functions: `get_base_id`, `parse_query`, `rrf_fuse`.
