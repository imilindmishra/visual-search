import streamlit as st
import requests
from PIL import Image
import base64
import io

st.set_page_config(page_title="Visual Product Search", layout="wide")
st.title("Visual Product Search")

col1, col2 = st.columns([1, 2])

with col1:
    input_method = st.radio("Input method", ["Upload", "Camera"])
    
    if input_method == "Upload":
        uploaded = st.file_uploader("Upload a product photo", type=["jpg", "jpeg", "png"])
        image_data = uploaded
    else:
        image_data = st.camera_input("Point camera at product")
    
    text_query = st.text_input("Add context (optional)")
    intent = st.radio("What do you want?", ["recommend", "compare", "authentic"])
    use_yolo = st.checkbox("Auto-crop product (YOLO)", value=False)
    search_btn = st.button("Search", type="primary")

with col2:
    if search_btn and image_data:
        with st.spinner("Searching..."):
            response = requests.post(
                "http://localhost:8000/search",
                files={"file": ("photo.jpg", image_data.getvalue(), "image/jpeg")},
                data={
                    "text_query": text_query,
                    "intent": intent,
                    "use_yolo": str(use_yolo).lower(),
                },
            )

        if response.status_code == 200:
            data = response.json()

            st.subheader("AI Response")
            st.write(data["answer"])

            st.subheader("Top Matches")
            for p in data["products"]:
                with st.expander(f"{p['name']} — score: {p['score']}"):
                    col_img, col_info = st.columns([1, 2])
                    with col_img:
                        if "image_b64" in p and p["image_b64"]:
                            img_bytes = base64.b64decode(p["image_b64"])
                            st.image(img_bytes, width=300)
                    with col_info:
                        st.write(f"Category: {p['category']}")
                        st.write(f"Color: {p.get('color', 'N/A')}")
                        st.write(f"Price: ₹{p['price']}")
                        st.write(p["description"])
        else:
            st.error(f"Error: {response.status_code}")