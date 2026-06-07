"""Central config — all tunables and constants for backend + ingestion live here."""

import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "products"
TOP_K = 5
CANDIDATE_POOL = 20
BATCH_SIZE = 100
IMAGE_QUALITY = 60
RRF_K = 60
IMAGE_WEIGHT = 0.5
TEXT_WEIGHT = 0.5

COLORS = [
    "blue", "red", "black", "white", "green", "yellow",
    "cream", "grey", "gray", "brown", "pink", "orange",
    "navy", "beige", "purple", "denim", "olive"
]

# Matches the actual category2 values ingested into Qdrant from deepfashion-inshop —
# there is no separate tshirts/jeans/polos bucket in the data (verified via scroll on /products).
CATEGORIES = ["pants", "shorts", "sweatshirts", "shirts", "sweaters",
              "jackets", "denim", "suiting", "tees"]
