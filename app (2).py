import os
os.environ["STREAMLIT_WATCHDOG_IGNORE"] = "true"  # Prevent inotify errors

import streamlit as st
from transformers import Blip2Processor, Blip2ForConditionalGeneration, pipeline
from PIL import Image
import torch

st.title("🖼️ BLIP2 Vision App")

# ----------------------
# Lazy model loaders
# ----------------------
@st.cache_resource
def load_caption_model():
    st.info("Loading Caption model... ⏳")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", device_map={"": torch.device("cpu")}
    )
    st.success("✅ Caption model loaded!")
    return processor, model

@st.cache_resource
def load_vqa_model():
    st.info("Loading VQA model... ⏳")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-flan-t5-xl", device_map={"": torch.device("cpu")}
    )
    st.success("✅ VQA model loaded!")
    return processor, model

@st.cache_resource
def load_translation_models():
    st.info("Loading translation pipelines... ⏳")
    models = {
        "Hindi": pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi"),
        "French": pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr"),
        "Spanish": pipeline("translation", model="Helsinki-NLP/opus-mt-en-es"),
    }
    st.success("✅ Translation pipelines loaded!")
    return models

# ----------------------
# Functions
# ----------------------
def generate_caption_translate(image, target_lang, caption_processor, caption_model, translation_models):
    inputs = caption_processor(image, return_tensors="pt")
    out = caption_model.generate(**inputs, max_new_tokens=50)
    english_caption = caption_processor.decode(out[0], skip_special_tokens=True)

    if target_lang in translation_models:
        translated = translation_models[target_lang](english_caption)[0]["translation_text"]
    else:
        translated = "Translation not available"

    return english_caption, translated

def vqa(image, question, vqa_processor, vqa_model):
    inputs = vqa_processor(image, question, return_tensors="pt").to(vqa_model.device)
    out = vqa_model.generate(**inputs, max_new_tokens=100)
    answer = vqa_processor.decode(out[0], skip_special_tokens=True)
    return answer

# ----------------------
# Streamlit UI
# ----------------------
tab1, tab2 = st.tabs(["Caption + Translate", "Visual Question Answering (VQA)"])

with tab1:
    st.header("Caption + Translate")
    img = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    lang = st.selectbox("Translate To", ["Hindi", "French", "Spanish"])
    if st.button("Generate Caption & Translate") and img is not None:
        # Lazy-load caption model and translation pipelines
        caption_processor, caption_model = load_caption_model()
        translation_models = load_translation_models()

        image = Image.open(img).convert("RGB")
        eng_caption, trans_caption = generate_caption_translate(
            image, lang, caption_processor, caption_model, translation_models
        )

        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.text_area("English Caption", eng_caption)
        st.text_area(f"Translated Caption ({lang})", trans_caption)

with tab2:
    st.header("Visual Question Answering (VQA)")
    img_vqa = st.file_uploader("Upload Image for VQA", type=["png", "jpg", "jpeg"], key="vqa_img")
    question = st.text_input("Ask a Question about the Image")
    if st.button("Ask") and img_vqa is not None and question:
        # Lazy-load VQA model
        vqa_processor, vqa_model = load_vqa_model()

        image = Image.open(img_vqa).convert("RGB")
        answer = vqa(image, question, vqa_processor, vqa_model)

        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.text_area("Answer", answer)
