import logging

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchText

from config import (
    QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME,
    TOP_K, CANDIDATE_POOL, RRF_K, COLORS,
)

logger = logging.getLogger(__name__)
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def get_base_id(product_id: str) -> str:
    """
    Strips view/angle suffix from product ID.
    'MEN_Denim_id_00005724_01_3_back' → 'MEN_Denim_id_00005724'
    """
    parts = product_id.split("_")
    for i, part in enumerate(parts):
        if part == "id" and i + 1 < len(parts):
            return "_".join(parts[:i + 2])
    return product_id


def parse_query(text_query: str) -> dict:
    """Extract structured constraints (color, category, gender) from a free text query."""
    result = {"color": None, "category": None, "gender": None}

    query_lower = text_query.lower()

    for color in COLORS:
        if color in query_lower:
            result["color"] = color
            break

    # Targets must be real category2 values in Qdrant (see config.CATEGORIES) —
    # the dataset has no separate tshirts/jeans/polos buckets, so map to the closest real one.
    category_map = {
        "tshirt": "sweatshirts", "t-shirt": "sweatshirts", "shirt": "shirts",
        "jeans": "denim", "jean": "denim", "denim": "denim",
        "pant": "pants", "trouser": "pants",
        "short": "shorts", "jacket": "jackets",
        "sweater": "sweaters", "hoodie": "sweatshirts",
        "polo": "shirts"
    }
    for keyword, category in category_map.items():
        if keyword in query_lower:
            result["category"] = category
            break

    if "men" in query_lower and "women" not in query_lower:
        result["gender"] = "men"
    elif "women" in query_lower or "woman" in query_lower:
        result["gender"] = "women"

    return result


def build_filter(parsed: dict) -> Filter | None:
    """Build a Qdrant payload filter from parsed query constraints (color, category as hard filters)."""
    conditions = []
    if parsed.get("color"):
        conditions.append(FieldCondition(key="color", match=MatchText(text=parsed["color"])))
    if parsed.get("category"):
        conditions.append(FieldCondition(key="category", match=MatchText(text=parsed["category"])))
    if not conditions:
        return None
    return Filter(must=conditions)


def rrf_fuse(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """
    rankings: list of lists of product IDs ordered best to worst
    Returns: dict of {product_id: rrf_score}
    """
    scores = {}
    for ranking in rankings:
        for rank, pid in enumerate(ranking):
            scores[pid] = scores.get(pid, 0) + 1.0 / (k + rank + 1)
    return scores


def _run_query(vector: list, using: str, query_filter: Filter | None, limit: int):
    """Query one named vector space in Qdrant, returning raw scored points."""
    return client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        using=using,
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
    ).points


def search_products(
    image_vector: list = None,
    text_vector: list = None,
    top_k: int = TOP_K,
    text_query: str = "",
) -> tuple[list[dict], dict]:
    """
    Run image + text vector search, fuse with RRF, dedupe by base product id,
    and return the top_k products plus the parsed query (for response transparency).
    Hard filters (color/category) are applied first; if they yield nothing, retry unfiltered.
    """
    parsed = parse_query(text_query)
    payload_filter = build_filter(parsed)

    payload_by_id = {}
    rankings = []

    if image_vector:
        image_results = _run_query(image_vector, "image", payload_filter, CANDIDATE_POOL)
        rankings.append([r.id for r in image_results])
        for r in image_results:
            payload_by_id[r.id] = r.payload

    if text_vector:
        text_results = _run_query(text_vector, "text", payload_filter, CANDIDATE_POOL)
        rankings.append([r.id for r in text_results])
        for r in text_results:
            payload_by_id.setdefault(r.id, r.payload)

    # Filtered search returned nothing — retry unfiltered
    if payload_filter and not payload_by_id:
        logger.info("filtered search empty for parsed=%s — falling back to unfiltered", parsed)
        return search_products(
            image_vector=image_vector,
            text_vector=text_vector,
            top_k=top_k,
            text_query="",
        )

    fused = rrf_fuse(rankings)

    # Dedupe by base product id, keeping the highest-scoring variant per product
    best_by_base = {}
    for pid, score in fused.items():
        payload = payload_by_id[pid]
        base_id = get_base_id(payload.get("product_id", pid))
        if base_id not in best_by_base or score > best_by_base[base_id]["score"]:
            best_by_base[base_id] = {"score": round(score, 6), **payload}

    results = sorted(best_by_base.values(), key=lambda x: x["score"], reverse=True)
    return results[:top_k], parsed