from func1 import detect_fire, classify_fire_presence
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import google.generativeai as genai
import os
from process_image import process_image_and_count
import torch
from collections import defaultdict
import tempfile

st.image('logo.png', width=300)
st.title("Fire Detection App")

model = st.radio("Choose Model", 
                 ("Vision LLMs", "Object Detection")
                 )

# File uploader for uploading an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    # if model == "Traditional Computer Vision": 
        
    #     with st.spinner('Processing...'):

    #         # Convert the uploaded file to an OpenCV image
    #         image = np.array(Image.open(uploaded_file))
    #         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    #         # Run fire detection
    #         fire_mask = detect_fire(image)
    #         fire_present = classify_fire_presence(fire_mask)

    #         # Display the uploaded image
    #         st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    #         # Display fire detection result
    #         if fire_present:
    #             st.error("🔥 Fire Detected!")
    #         else:
    #             st.success("✅ No Fire Detected.")

    #         # Display the fire mask
    #         st.image(fire_mask, caption="Fire Mask", use_column_width=True, clamp=True)
    
    if model == "Vision LLMs":

        genai.configure(api_key=os.environ["GEMINI_API_KEY"])

        # model_choice = st.selectbox("Choose the model:", ("Model F", "Model P"))

        model_choice = 'Model F'
        
        model_choice = 'gemini-1.5-Flash' if model_choice=='Model F' else 'gemini-1.5-Pro'

        custom_prompt = "Is there fire and smoke in the image?"

        image = Image.open(uploaded_file)

        st.image(image, caption='Uploaded Image', use_column_width=True)

        with st.spinner('Processing...'):

            model = genai.GenerativeModel(model_name=f'models/{model_choice.lower()}')

            response = model.generate_content([custom_prompt, image])

            st.write("Response:")

            st.write(response.text)

    else:
        selected_classes = ['Fire']
        classNames = ['Fire', 'default', 'smoke']
        cuda_available = torch.cuda.is_available()
        device = 'cuda:0' if cuda_available else 'cpu'
        half = cuda_available 
        iou = 0.6
        conf = 0.15
        imgsz = 600
        vid_stride = 2
        augment = False 
        run_dir = "runs/temp"
        os.makedirs(run_dir, exist_ok=True)
        selected_model = 'yolov8n datav1 40epochs.pt'

        if "tracked_objects" not in st.session_state:
            st.session_state["tracked_objects"] = defaultdict(lambda: defaultdict(int))

        with st.spinner('Processing...'):

            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file.type.split("/")[-1]}')

            tfile.write(uploaded_file.getvalue())

            file_path = tfile.name
            
            object_counts, output_path = process_image_and_count(file_path, selected_model, classNames, run_dir, iou=iou, conf=conf, imgsz=imgsz, augment=augment, device=device)
            st.session_state["tracked_objects"] = object_counts
            st.image(output_path)
