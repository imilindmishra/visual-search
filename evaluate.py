"""
Offline evaluation — run this to measure Recall@K.
Tests whether correct category appears in top-5 results.
Usage: python evaluate.py
"""

import requests
import json

TEST_CASES = [
    {"text_query": "blue tshirt", "expected_category": "sweatshirts"},
    {"text_query": "black jeans", "expected_category": "denim"},
    {"text_query": "white shirt men", "expected_category": "shirts"},
    {"text_query": "navy pants", "expected_category": "pants"},
    {"text_query": "grey hoodie", "expected_category": "sweatshirts"},
    {"text_query": "brown shorts", "expected_category": "shorts"},
    {"text_query": "green jacket", "expected_category": "jackets"},
    {"text_query": "red polo", "expected_category": "shirts"},
    {"text_query": "cream sweater", "expected_category": "sweaters"},
    {"text_query": "black denim", "expected_category": "denim"},
]


def evaluate():
    """Run all TEST_CASES against the live /search endpoint and print Recall@5."""
    hits = 0
    results = []

    for case in TEST_CASES:
        # Use a blank 224x224 white image for text-only evaluation
        import numpy as np
        from PIL import Image
        import io
        blank = Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8) * 255)
        buf = io.BytesIO()
        blank.save(buf, format="JPEG")
        buf.seek(0)

        response = requests.post(
            "http://localhost:8000/search",
            files={"file": ("blank.jpg", buf, "image/jpeg")},
            data={
                "text_query": case["text_query"],
                "intent": "recommend",
                "use_yolo": "false",
            },
        )

        if response.status_code != 200:
            results.append({**case, "hit": False, "returned": "error"})
            continue

        data = response.json()
        returned_categories = [p["category"].split("/")[1].strip()
                               for p in data["products"]]
        hit = case["expected_category"] in returned_categories
        if hit:
            hits += 1

        results.append({
            **case,
            "hit": hit,
            "returned_categories": returned_categories,
        })

    recall_at_5 = hits / len(TEST_CASES)

    print(f"\nRecall@5: {hits}/{len(TEST_CASES)} = {recall_at_5:.1%}\n")
    for r in results:
        status = "✅" if r["hit"] else "❌"
        print(f"{status} '{r['text_query']}' → {r['returned_categories']}")

    return recall_at_5


if __name__ == "__main__":
    evaluate()
