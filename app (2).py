# app.py
# import gradio as gr
# from transformers import BlipProcessor, BlipForConditionalGeneration
# from gtts import gTTS
# import io
# from PIL import Image

# # -------------------------------
# # Load BLIP-base model (lighter version)
# # -------------------------------
# processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
# model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# # -------------------------------
# # Generate caption function
# # -------------------------------
# # def generate_caption_tts(image):
# #     caption = generate_caption(model, processor, image)
# #     audio_file = text_to_audio_file(caption)
# #     return caption, audio_file  # return file path, not BytesIO


# # -------------------------------
# # Convert text to speech using gTTS
# # -------------------------------
# import tempfile
# import pyttsx3

# def text_to_audio_file(text):
#     # Create a temporary file
#     tmp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
#     tmp_path = tmp_file.name
#     tmp_file.close()

#     engine = pyttsx3.init()
#     engine.save_to_file(text, tmp_path)
#     engine.runAndWait()

#     return tmp_path

# def generate_caption_from_image(model, processor, image):
#     # image: PIL.Image
#     inputs = processor(images=image, return_tensors="pt")
#     out = model.generate(**inputs)
#     caption = processor.decode(out[0], skip_special_tokens=True)
#     return caption
# # -------------------------------
# # Gradio interface: Caption + Audio
# # -------------------------------
# def generate_caption_tts(image):
#     caption = generate_caption_from_image(model, processor, image)  # uses global model/processor
#     # audio_file = text_to_audio_file(caption)
#     return caption 



# interface = gr.Interface(
#     fn=generate_caption_tts,
#     inputs=gr.Image(type="numpy"),
#     outputs=[gr.Textbox(label="Generated Caption")],
#     title="Image Captioning for Visually Impaired",
#     description="Upload an image, get a caption and audio description."
# )


# interface.launch()
# # demo.launch(share=True)

import gradio as gr
from transformers import Blip2Processor, Blip2ForConditionalGeneration, pipeline
from PIL import Image
import torch

# ----------------------
# Load BLIP2 for Captioning
# ----------------------
caption_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
caption_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

# ----------------------
# Load BLIP2 for VQA
# ----------------------
vqa_processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
vqa_model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl", torch_dtype=torch.float16, device_map="auto"
)

# ----------------------
# Translation pipelines
# ----------------------
translation_models = {
    "Hindi": pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi"),
    "French": pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr"),
    "Spanish": pipeline("translation", model="Helsinki-NLP/opus-mt-en-es"),
}

# ----------------------
# Caption + Translate Function
# ----------------------
def generate_caption_translate(image, target_lang):
    inputs = caption_processor(image, return_tensors="pt")
    out = caption_model.generate(**inputs, max_new_tokens=50)
    english_caption = caption_processor.decode(out[0], skip_special_tokens=True)

    # Translate if chosen
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

demo.launch(share="true")





 