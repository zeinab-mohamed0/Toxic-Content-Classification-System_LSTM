import streamlit as st
from PIL import Image

from image_captioning import generate_caption
from text_classifier import predict_label
from database import save_result

st.set_page_config(page_title="AI Content Classifier", layout="centered")

st.title("🧠 AI Image Caption & Toxicity Classifier")

# upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Analyze 🔍"):
        with st.spinner("Processing..."):

            # 1. caption
            caption = generate_caption(image)

            # 2. classification
            label = predict_label(caption)

            # 3. display
            st.subheader("📌 Caption:")
            st.write(caption)

            st.subheader("Predicted Label:")
            if label == "Safe":
                st.success(label)
            else:
                st.error(label)
            # 4. save
            save_result(caption, label)

            st.info("Saved to database ✅")

            if st.button("Clear"):
                st.experimental_rerun()