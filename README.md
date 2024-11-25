# adversarial-analyzer

Adversarial Image Analyzer

Explore the fascinating world of adversarial attacks on deep learning models with the Adversarial Image Analyzer. This interactive tool showcases how small perturbations can drastically affect AI predictions.

Live Demo:
Adversarial Image Analyzer: https://adversarial-analyzer.streamlit.app/

Features

1. Upload images (.jpg, .jpeg, .png) for analysis.
2. Adjust perturbation strength (epsilon) using a slider.
3. View side-by-side comparisons of original and adversarial images.
4. Inspect model predictions and confidence scores for both images.

How to Use
1. Upload an Image: Select an image file using the uploader.
2. Adjust Epsilon: Use the slider to control the strength of perturbations.
3. Generate Results: Click “Submit” to generate and view results.

The app will display:
1. Original and adversarial images.
2. Predictions and confidence scores for both images.

Technical Details
Model: Pre-trained ResNet-50 from PyTorch’s torchvision.
Method: Fast Gradient Sign Method (FGSM) to generate adversarial examples.
Backend: Flask API for adversarial image processing.
Frontend: Streamlit for an interactive UI.
