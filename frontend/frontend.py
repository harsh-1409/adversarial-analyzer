import streamlit as st
import requests
from PIL import Image
import io
import os
import base64
from flask import Flask, request, jsonify
from torchvision import models, transforms
from torchvision.models import resnet50, ResNet50_Weights
import torch
import numpy as np
import json
import logging
from threading import Thread

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Flask Backend
app = Flask(__name__)
upload_folder = 'uploads'
os.makedirs(upload_folder, exist_ok=True)
app.config['UPLOAD_FOLDER'] = upload_folder

# Load pretrained model
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
model.eval()

# Preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to generate adversarial example
def generate_adversarial_example(model, img_tensor, epsilon=0.3):
    img_tensor.requires_grad = True
    output = model(img_tensor)
    _, target_class = output.max(1)
    loss = torch.nn.CrossEntropyLoss()(output, target_class)
    model.zero_grad()
    loss.backward()

    adv_img_tensor = img_tensor + epsilon * img_tensor.grad.sign()
    adv_img_tensor = torch.clamp(adv_img_tensor, 0, 1)

    return adv_img_tensor

# Upload endpoint
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        app.logger.info("File received and processing started.")
        if 'file' not in request.files:
            return jsonify({'error': 'No file found'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        img = Image.open(file_path).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0)
        epsilon = float(request.args.get('epsilon', 0.3))

        adv_img_tensor = generate_adversarial_example(model, img_tensor, epsilon)

        original_img = img_tensor.squeeze().permute(1, 2, 0).detach().numpy()
        adv_img = adv_img_tensor.squeeze().permute(1, 2, 0).detach().numpy()

        output = model(img_tensor)
        original_class = torch.argmax(output, dim=1).item()

        adv_output = model(adv_img_tensor)
        adv_class = torch.argmax(adv_output, dim=1).item()

        labels_path = './data/imagenet-simple-labels.json'
        with open(labels_path) as f:
            labels = json.load(f)

        original_label = labels[original_class]
        adv_label = labels[adv_class]

        adv_img = (adv_img * 255).astype(np.uint8)
        adv_img_pil = Image.fromarray(adv_img)

        buffer = io.BytesIO()
        adv_img_pil.save(buffer, format='PNG')
        adv_img_encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return jsonify({
            "original_prediction": {
                "class": int(original_class),
                "label": original_label,
                "confidence": float(torch.softmax(output, dim=1).max().item())
            },
            "adversarial_prediction": {
                "class": int(adv_class),
                "label": adv_label,
                "confidence": float(torch.softmax(adv_output, dim=1).max().item()),
                "adversarial_image": adv_img_encoded
            }
        })
    except Exception as e:
        app.logger.error(f"Error processing the image: {e}")
        return jsonify({'error': 'Error processing the image'}), 500

# Run Flask server in a thread
def run_flask():
    app.run(debug=False, use_reloader=False)

Thread(target=run_flask).start()

# Streamlit Frontend
st.title("Adversarial Image Analyzer")
st.markdown("Upload an image to generate and analyze adversarial examples.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
epsilon = st.slider('Select Perturbation Strength (Epsilon):', 0.0, 1.0, 0.3, step=0.05, help="Epsilon is the strength of the perturbation added to the image.")

def decode_image(image_data):
    return base64.b64decode(image_data)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if st.button("Submit"):
        st.write("Generating adversarial example...")
        try:
            files = {"file": uploaded_file.getvalue()}
            response = requests.post(
                f"http://127.0.0.1:5000/upload?epsilon={epsilon}",
                files=files
            )

            if response.status_code == 200:
                result = response.json()
                decoded_image = decode_image(result['adversarial_prediction']['adversarial_image'])

                st.write("### Results:")
                st.write("**Original Prediction:**")
                st.write(f"Class: {result['original_prediction']['label']}")
                st.write(f"Confidence: {result['original_prediction']['confidence']:.2f}")

                st.write("**Adversarial Prediction:**")
                st.write(f"Class: {result['adversarial_prediction']['label']}")
                st.write(f"Confidence: {result['adversarial_prediction']['confidence']:.2f}")

                adv_img = Image.open(io.BytesIO(decoded_image))

                col1, col2 = st.columns(2)
                with col1:
                    st.image(uploaded_file, caption="Original Image", use_container_width=True)
                with col2:
                    st.image(adv_img, caption="Adversarial Image", use_container_width=True)
            else:
                st.error("Error processing the image. Please try again.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.info("Please upload an image to begin.")