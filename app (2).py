import gradio as gr
from transformers import Blip2Processor, Blip2ForConditionalGeneration, pipeline
from PIL import Image
import torch
import streamlit as st

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
# Load All Models with Spinner
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
# Gradio UI
# ----------------------
with gr.Blocks(title="BLIP2 Vision App") as demo:
    gr.Markdown("## 🖼️ BLIP2: Image Captioning + Translation + Question Answering")

    with gr.Tab("Caption + Translate"):
        with gr.Row():
            img_in = gr.Image(type="pil", label="Upload Image")
            lang_in = gr.Dropdown(["Hindi", "French", "Spanish"], label="Translate To")
        eng_out = gr.Textbox(label="English Caption")
        trans_out = gr.Textbox(label="Translated Caption")
        btn1 = gr.Button("Generate Caption & Translate")
        btn1.click(generate_caption_translate, inputs=[img_in, lang_in], outputs=[eng_out, trans_out])

    with gr.Tab("Visual Question Answering (VQA)"):
        with gr.Row():
            img_vqa = gr.Image(type="pil", label="Upload Image")
            q_in = gr.Textbox(label="Ask a Question about the Image")
        ans_out = gr.Textbox(label="Answer")
        btn2 = gr.Button("Ask")
        btn2.click(vqa, inputs=[img_vqa, q_in], outputs=ans_out)

demo.launch()
