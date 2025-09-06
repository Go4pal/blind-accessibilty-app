# import os
# os.environ["STREAMLIT_WATCHDOG_IGNORE"] = "true"  # Prevent inotify errors

# import streamlit as st
# from transformers import Blip2Processor, Blip2ForConditionalGeneration
# from PIL import Image
# import torch

# st.title("🖼️ BLIP2 Vision App")

# # ----------------------
# # Lazy model loaders
# # ----------------------
# @st.cache_resource
# def load_caption_model():
#     st.info("Loading Caption model... ⏳")
#     processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
#     model = Blip2ForConditionalGeneration.from_pretrained(
#         "Salesforce/blip2-opt-2.7b", device_map={"": torch.device("cpu")}
#     )
#     st.success("✅ Caption model loaded!")
#     return processor, model

# @st.cache_resource
# def load_vqa_model():
#     st.info("Loading VQA model... ⏳")
#     processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
#     model = Blip2ForConditionalGeneration.from_pretrained(
#         "Salesforce/blip2-flan-t5-xl", device_map={"": torch.device("cpu")}
#     )
#     st.success("✅ VQA model loaded!")
#     return processor, model

# # ----------------------
# # Functions
# # ----------------------
# def generate_caption(image, caption_processor, caption_model):
#     inputs = caption_processor(image, return_tensors="pt")
#     out = caption_model.generate(**inputs, max_new_tokens=50)
#     eng_caption = caption_processor.decode(out[0], skip_special_tokens=True)
#     return eng_caption

# def vqa(image, question, vqa_processor, vqa_model):
#     inputs = vqa_processor(image, question, return_tensors="pt").to(vqa_model.device)
#     out = vqa_model.generate(**inputs, max_new_tokens=100)
#     answer = vqa_processor.decode(out[0], skip_special_tokens=True)
#     return answer

# # ----------------------
# # Streamlit UI
# # ----------------------
# tab1, tab2 = st.tabs(["Caption Only", "Visual Question Answering (VQA)"])

# with tab1:
#     st.header("Caption Only")
#     img = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
#     if st.button("Generate Caption") and img is not None:
#         # Lazy-load caption model
#         caption_processor, caption_model = load_caption_model()

#         image = Image.open(img).convert("RGB")
#         eng_caption = generate_caption(image, caption_processor, caption_model)

#         st.image(image, caption="Uploaded Image", use_column_width=True)
#         st.text_area("English Caption", eng_caption)

# with tab2:
#     st.header("Visual Question Answering (VQA)")
#     img_vqa = st.file_uploader("Upload Image for VQA", type=["png", "jpg", "jpeg"], key="vqa_img")
#     question = st.text_input("Ask a Question about the Image")
#     if st.button("Ask") and img_vqa is not None and question:
#         # Lazy-load VQA model
#         vqa_processor, vqa_model = load_vqa_model()

#         image = Image.open(img_vqa).convert("RGB")
#         answer = vqa(image, question, vqa_processor, vqa_model)

#         st.image(image, caption="Uploaded Image", use_column_width=True)
#         st.text_area("Answer", answer)




import streamlit as st
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image

# ----------------------
# Load Smaller BLIP2 Model (cached)
# ----------------------
@st.cache_resource
def load_model():
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-350m")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-350m",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    return processor, model

st.title("🖼️ BLIP2: Fast Image Captioning")

processor, model = load_model()

# ----------------------
# Upload Image
# ----------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Caption"):
        with st.spinner("⚡ Generating caption..."):
            inputs = processor(images=image, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=50)
            caption = processor.decode(output_ids[0], skip_special_tokens=True)

        if caption.strip():
            st.success(f"**Caption:** {caption}")
        else:
            st.error("⚠️ No caption generated. Try another image.")
