import streamlit as st
import requests
from PIL import Image
import io
import base64

st.title('Adversarial Image Generation')
st.markdown("Upload an image to generate and analyze adversarial examples.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", 'jpeg', 'png'])

epsilon = st.slider('Select Perturbation Strength (Epsilon):', min_value=0.0, max_value=1.0, value=0.3, step=0.05, help="Epsilon is the strength of the perturbation added to the original image to generate the adversarial example.")

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Add a "Submit" button
    if st.button("Submit"):
        st.write("Generating adversarial example...")
        # Decode the adversarial image from the response
        def decode_image(image_data):
            return base64.b64decode(image_data)

        
        try:
            # Send the file to the backend when "Submit" is clicked
            files = {"file": uploaded_file.getvalue()}
            response = requests.post(
                f"http://127.0.0.1:5000/upload?epsilon={epsilon}",
                files=files
            )

            # Process the backend response
            if response.status_code == 200:
                result = response.json()
                # Add the decoding function call
                decoded_image = decode_image(result['adversarial_prediction']['adversarial_image'])
                st.write("### Results:")
                st.write("**Original Prediction:** The model's classification and confidence score for the unmodified image.")
                st.write(f"Class: {result['original_prediction']['label']}")
                st.write(f"Confidence: {result['original_prediction']['confidence']:.2f}")

                st.write("**Adversarial Prediction:** The model's classification and confidence score for the adversarial image after adding perturbations.")
                st.write(f"Class: {result['adversarial_prediction']['label']}")
                st.write(f"Confidence: {result['adversarial_prediction']['confidence']:.2f}")
                st.write("### Adversarial Image:")
                #adv_img = Image.open(io.BytesIO(bytearray(result['adversarial_prediction']['adversarial_image'])))
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