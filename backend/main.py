from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import io, anthropic, os, logging
from dotenv import load_dotenv

from vision import encode_image, encode_text, detect_and_crop
from rag import search_products
from prompts import build_prompt
from config import TOP_K

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)
app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


class ProductResult(BaseModel):
    """Single retrieved product as returned to the client."""
    name: str
    category: str
    color: str
    price: float
    score: float
    description: str
    image_b64: str | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "MEN Denim id 00005724",
                "category": "MEN / denim",
                "color": "blue",
                "price": 1234.56,
                "score": 0.0421,
                "description": "men denim blue slim fit jeans",
                "image_b64": "/9j/4AAQSkZJRg... (truncated)",
            }
        }
    }


class SearchResponse(BaseModel):
    """Response payload for the /search endpoint."""
    products: list[ProductResult]
    answer: str
    intent: str
    query_parsed: dict

    model_config = {
        "json_schema_extra": {
            "example": {
                "products": [ProductResult.model_config["json_schema_extra"]["example"]],
                "answer": "I'd recommend the MEN Denim id 00005724 — closest visual match "
                          "and matches your 'blue jeans' query at a reasonable price point.",
                "intent": "recommend",
                "query_parsed": {"color": "blue", "category": "denim", "gender": "men"},
            }
        }
    }


@app.get("/health")
def health():
    """Liveness check."""
    return {"status": "ok"}


@app.post(
    "/search",
    response_model=SearchResponse,
    summary="Search the catalog by image (+ optional text)",
    description=(
        "Multipart form upload. Encodes the image (and text query, if given) with CLIP, "
        "retrieves visually + semantically similar products from Qdrant, fuses rankings with "
        "RRF, dedupes by base product id, and asks Claude to advise on the results.\n\n"
        "**Example request (curl):**\n"
        "```bash\n"
        "curl -X POST http://localhost:8000/search \\\n"
        "  -F \"file=@product.jpg\" \\\n"
        "  -F \"text_query=blue jeans for men\" \\\n"
        "  -F \"intent=recommend\" \\\n"
        "  -F \"use_yolo=true\"\n"
        "```"
    ),
)
async def search(
    file: UploadFile = File(..., description="Product photo (jpg/jpeg/png)"),
    text_query: str = Form("", description="Optional free-text context", examples=["blue jeans for men"]),
    intent: str = Form(
        "recommend",
        description="What kind of advice to ask Claude for",
        examples=["recommend", "compare", "authentic"],
    ),
    use_yolo: bool = Form(False, description="Auto-crop the product region with YOLO before encoding", examples=[False]),
):
    """Search the catalog by image (+ optional text), rank with RRF, and ask Claude to advise on the results."""
    logger.info("search request: text_query=%r intent=%r use_yolo=%s", text_query, intent, use_yolo)

    # 1. Load image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # 2. Optional YOLO crop — CLIP/YOLO inference is blocking torch work, run off the event loop
    if use_yolo:
        image = await run_in_threadpool(detect_and_crop, image)

    # 3. Encode image
    img_vector = await run_in_threadpool(encode_image, image)

    # 4. Encode text if provided
    txt_vector = await run_in_threadpool(encode_text, text_query) if text_query.strip() else None

    # 5. Retrieve from Qdrant (image + text search fused via RRF, deduped, hard-filtered)
    try:
        products, query_parsed = await run_in_threadpool(
            search_products,
            image_vector=img_vector,
            text_vector=txt_vector,
            top_k=TOP_K,
            text_query=text_query,
        )
    except Exception:
        logger.exception("Qdrant search failed")
        raise HTTPException(status_code=503, detail="Product search is temporarily unavailable")

    logger.info("search retrieved %d products, query_parsed=%s", len(products), query_parsed)

    # 6. Build prompt and call LLM
    prompt = build_prompt(products, intent)
    try:
        response = await run_in_threadpool(
            claude.messages.create,
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception:
        logger.exception("Claude call failed")
        raise HTTPException(status_code=502, detail="AI advisor is temporarily unavailable")
    llm_answer = response.content[0].text

    return SearchResponse(
        products=products,
        answer=llm_answer,
        intent=intent,
        query_parsed=query_parsed,
    )