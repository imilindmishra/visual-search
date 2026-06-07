import clip
import torch
import base64
import io
from datasets import load_dataset
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, NamedVector
import uuid

from backend.config import (
    QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, BATCH_SIZE, IMAGE_QUALITY,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Two named vector spaces — one for image, one for text
if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "image": VectorParams(size=512, distance=Distance.COSINE),
            "text": VectorParams(size=512, distance=Distance.COSINE),
        },
    )

dataset = load_dataset("Marqo/deepfashion-inshop", split="data[:5000]")

points = []
for i, item in enumerate(dataset):
    image = item["image"].convert("RGB")

    # Image embedding
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_vector = model.encode_image(image_input).squeeze().cpu().numpy().tolist()

    # Text embedding — encode description
    desc = item["text"] if item["text"] else f"{item['category1']} {item['category2']} {item['color']}"
    desc = desc[:300]
    text_tokens = clip.tokenize([desc], truncate=True).to(device)
    with torch.no_grad():
        text_vector = model.encode_text(text_tokens).squeeze().cpu().numpy().tolist()

    # Image for display
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=IMAGE_QUALITY)
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    payload = {
        "name": item["item_ID"].replace("_", " "),
        "category": f"{item['category1']} / {item['category2']}",
        "color": item["color"] or "unknown",
        "description": desc,
        "product_id": item["item_ID"],
        "price": round(500 + (i * 7.3) % 4500, 2),
        "image_b64": img_base64,
    }

    points.append(PointStruct(
        id=str(uuid.uuid4()),
        vector={
            "image": image_vector,
            "text": text_vector,
        },
        payload=payload,
    ))

    if i % 100 == 0:
        print(f"Embedded {i}/5000")

# Batch upsert
for i in range(0, len(points), BATCH_SIZE):
    batch = points[i:i + BATCH_SIZE]
    client.upsert(collection_name=COLLECTION_NAME, points=batch)
    print(f"Uploaded {min(i + BATCH_SIZE, len(points))}/5000")

print("Catalog ingestion complete.")