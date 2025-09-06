import os
os.environ["STREAMLIT_WATCHDOG_IGNORE"] = "true" 
import streamlit as st
from transformers import Blip2Processor, Blip2ForConditionalGeneration, pipeline
from PIL import Image
import torch

# ----------------------
# Cached Model Loaders
# ----------------------
@st.cache_resource
def load_caption_model():
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
    return processor, model

@st.cache_resource
def load_vqa_model():
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-flan-t5-xl", torch_dtype=torch.float16, device_map="auto"
    )
    return processor, model

@st.cache_resource
def load_translation_models():
    return {
        "Hindi": pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi"),
        "French": pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr"),
        "Spanish": pipeline("translation", model="Helsinki-NLP/opus-mt-en-es"),
    }

# ----------------------
# Load Models with Spinner
# ----------------------
with st.spinner("Loading BLIP2 models... please wait ⏳"):
    caption_processor, caption_model = load_caption_model()
    vqa_processor, vqa_model = load_vqa_model()
    translation_models = load_translation_models()

st.success("✅ Models are ready!")

# ----------------------
# Caption + Translate Function
# ----------------------
def generate_caption_translate(image, target_lang):
    inputs = caption_processor(image, return_tensors="pt")
    out = caption_model.generate(**inputs, max_new_tokens=50)
    english_caption = caption_processor.decode(out[0], skip_special_tokens=True)

    if target_lang in translation_models:
        translated = translation_models[target_lang](english_caption)[0]['translation_text']
    else:
        translated = "Translation not available"

    return english_caption, translated

# ----------------------
# VQA Function
# ----------------------
def vqa(image, question):
    inputs = vqa_processor(image, question, return_tensors="pt").to(vqa_model.device)
    out = vqa_model.generate(**inputs, max_new_tokens=100)
    answer = vqa_processor.decode(out[0], skip_special_tokens=True)
    return answer

# ----------------------
# Streamlit UI
# ----------------------
st.title("🖼️ BLIP2 Vision App")
tab1, tab2 = st.tabs(["Caption + Translate", "Visual Question Answering (VQA)"])

with tab1:
    st.header("Caption + Translate")
    img = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    lang = st.selectbox("Translate To", ["Hindi", "French", "Spanish"])
    if st.button("Generate Caption & Translate") and img is not None:
        image = Image.open(img).convert("RGB")
        eng_caption, trans_caption = generate_caption_translate(image, lang)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.text_area("English Caption", eng_caption)
        st.text_area(f"Translated Caption ({lang})", trans_caption)

with tab2:
    st.header("Visual Question Answering (VQA)")
    img_vqa = st.file_uploader("Upload Image for VQA", type=["png", "jpg", "jpeg"], key="vqa_img")
    question = st.text_input("Ask a Question about the Image")
    if st.button("Ask") and img_vqa is not None and question:
        image = Image.open(img_vqa).convert("RGB")
        answer = vqa(image, question)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.text_area("Answer", answer)
