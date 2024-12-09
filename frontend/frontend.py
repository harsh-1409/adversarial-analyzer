import streamlit as st
import requests
from PIL import Image
import io
import base64

# Streamlit App Title
st.set_page_config(
    page_title="Adversarial Image Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header Section
st.title("🔍 Adversarial Image Analyzer")
st.markdown(
    "Use this tool to generate adversarial examples and analyze the robustness of deep learning models."
)

# Sidebar Section
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

if uploaded_file:
    st.sidebar.image(
        uploaded_file,
        caption="Uploaded Image",
        use_container_width=True
    )

# Function to Decode the Adversarial Image
def decode_image(image_data):
    return base64.b64decode(image_data)

# Function to Display Results
def display_results(original_image, result):
    col1, col2 = st.columns(2)

    # Display Original Image and Predictions
    with col1:
        st.subheader("📷 Original Image")
        st.image(original_image, caption="Original Image", use_container_width=True)
        st.markdown(f"**Predicted Class:** {result['original_prediction']['label']}")
        st.markdown(f"**Confidence:** {result['original_prediction']['confidence']:.2f}")

    # Display Adversarial Image and Predictions
    with col2:
        st.subheader("⚠️ Adversarial Image")
        decoded_image = decode_image(result["adversarial_prediction"]["adversarial_image"])
        adv_img = Image.open(io.BytesIO(decoded_image))
        st.image(adv_img, caption="Adversarial Image", use_container_width=True)
        st.markdown(f"**Predicted Class:** {result['adversarial_prediction']['label']}")
        st.markdown(f"**Confidence:** {result['adversarial_prediction']['confidence']:.2f}")

# Main Section
if uploaded_file:
    # Display Uploaded Image
    
    st.markdown("### Review and Submit")

    if st.button("🛠 Generate Adversarial Example"):
        st.write("⏳ Generating adversarial example... Please wait.")
        try:
            # Prepare File for Upload
            files = {"file": uploaded_file.getvalue()}
            response = requests.post(
                f"http://127.0.0.1:5000/upload?epsilon={epsilon}",
                files=files
            )

            # Process Backend Response
            if response.status_code == 200:
                result = response.json()
                display_results(uploaded_file, result)
            else:
                st.error("❌ Error processing the image. Please try again.")
        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
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