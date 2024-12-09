import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import numpy as np
import base64
import io
import json

# Configure Streamlit App
st.set_page_config(
    page_title="Adversarial Image Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Pretrained Model
@st.cache_resource
def load_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.eval()
    return model

model = load_model()

# Preprocessing Transformations
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to Generate Adversarial Example
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

# Decode Base64 Image
def decode_image(image_data):
    return base64.b64decode(image_data)

# UI Sidebar for Input
st.sidebar.header("Upload Settings")
uploaded_file = st.sidebar.file_uploader(
    "Upload an image (JPG, JPEG, PNG):",
    type=["jpg", "jpeg", "png"]
)
epsilon = st.sidebar.slider(
    "Perturbation Strength (Epsilon):",
    min_value=0.0,
    max_value=1.0,
    value=0.3,
    step=0.05,
    help="Controls the strength of the perturbation added to the original image."
)

# Main Section
st.title("🔍 Adversarial Image Analyzer")
st.markdown(
    "Use this tool to generate adversarial examples and analyze the robustness of deep learning models."
)

if uploaded_file:
    st.sidebar.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Display Uploaded Image
    image = Image.open(uploaded_file).convert("RGB")
    #st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🛠 Generate Adversarial Example"):
        st.write("⏳ Generating adversarial example... Please wait.")

        # Preprocess the Image
        img_tensor = preprocess(image).unsqueeze(0)

        # Generate Adversarial Example
        adv_img_tensor = generate_adversarial_example(model, img_tensor, epsilon)

        # Convert Adversarial Image to Displayable Format
        adv_img = adv_img_tensor.squeeze().permute(1, 2, 0).detach().numpy()
        adv_img = (adv_img * 255).astype(np.uint8)
        labels_path = './data/imagenet-simple-labels.json'
        with open(labels_path) as f:
            labels = json.load(f)

        # Get Model Predictions
        output = model(img_tensor)
        original_class = torch.argmax(output, dim=1).item()
        original_class = labels[original_class]
        original_confidence =float(torch.softmax(output, dim=1).max().item())

        adv_output = model(adv_img_tensor)
        adv_class = torch.argmax(adv_output, dim=1).item()
        adv_class = labels[adv_class]
        adv_confidence = float(torch.softmax(adv_output, dim=1).max().item())

        # Display Results
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📷 Original Image")
            st.image(image, caption="Original Image", use_container_width=True)
            st.markdown(f"**Predicted Class:** {original_class}")
            st.markdown(f"**Class confidence:** {original_confidence:.2f}")

        with col2:
            st.subheader("⚠️ Adversarial Image")
            st.image(adv_img, caption="Adversarial Image", use_container_width=True)
            st.markdown(f"**Predicted Class:** {adv_class}")
            st.markdown(f"**Class confidence:** {adv_confidence:.2f}")

else:
    st.info("Please upload an image to get started!")

# Footer
st.markdown(
    """
    ---
    **Adversarial Image Analyzer**  
    Built with ❤️ using [Streamlit](https://streamlit.io/) and PyTorch.
    """
)