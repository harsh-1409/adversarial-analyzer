import streamlit as st
import requests
from PIL import Image
import io

st.title('Adversarial Image Generation')
st.markdown("Upload an image to generate and analyze adversarial examples.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", 'jpeg', 'png'])

epsilon = st.slider('Select Perturbation Strength (Epsilon):', min_value=0.0, max_value=1.0, value=0.5, step=0.1)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Add a "Submit" button
    if st.button("Submit"):
        st.write("Generating adversarial example...")
        
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
                st.write("### Results:")
                st.write("**Original Prediction:**")
                st.write(f"Class: {result['original_prediction']['label']}")
                st.write(f"Confidence: {result['original_prediction']['confidence']:.2f}")

                st.write("**Adversarial Prediction:**")
                st.write(f"Class: {result['adversarial_prediction']['label']}")
                st.write(f"Confidence: {result['adversarial_prediction']['confidence']:.2f}")
            else:
                st.error("Error processing the image. Please try again.")

        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.info("Please upload an image to begin.")