import clip
import torch
from PIL import Image
from ultralytics import YOLO
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
yolo_model = YOLO("yolov8n.pt")  # downloads automatically on first run

def encode_image(image: Image.Image) -> list[float]:
    """Convert a PIL image to a 512-dim CLIP vector."""
    img_input = clip_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        vector = clip_model.encode_image(img_input).squeeze().cpu().numpy()
    return vector.tolist()

def encode_text(text: str) -> list[float]:
    """Convert a text string to a 512-dim CLIP vector."""
    tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        vector = clip_model.encode_text(tokens).squeeze().cpu().numpy()
    return vector.tolist()

def detect_and_crop(image: Image.Image) -> Image.Image:
    """Run YOLO. If a product is found, return the cropped region. Else return original."""
    results = yolo_model(image)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return image
    # Take the highest-confidence detection
    best = boxes[boxes.conf.argmax()]
    x1, y1, x2, y2 = map(int, best.xyxy[0].tolist())
    return image.crop((x1, y1, x2, y2))

def merge_vectors(v1: list, v2: list) -> list:
    """Average two vectors element-by-element."""
    a = np.array(v1)
    b = np.array(v2)
    return ((a + b) / 2).tolist()