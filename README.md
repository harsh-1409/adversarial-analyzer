# adversarial-analyzer

Adversarial Image Analyzer

Explore the fascinating world of adversarial attacks on deep learning models with the Adversarial Image Analyzer. This interactive tool showcases how small perturbations can drastically affect AI predictions.

Live Demo:
Adversarial Image Analyzer: https://adversarial-analyzer.streamlit.app/

Features

• Upload images (.jpg, .jpeg, .png) for analysis.
• Adjust perturbation strength (epsilon) using a slider.
• View side-by-side comparisons of original and adversarial images.
• Inspect model predictions and confidence scores for both images.

How to Use
1. Upload an Image: Select an image file using the uploader.
2. Adjust Epsilon: Use the slider to control the strength of perturbations.
3. Generate Results: Click “Submit” to generate and view results.

The app will display:
• Original and adversarial images.
• Predictions and confidence scores for both images.

Technical Details
• Model: Pre-trained ResNet-50 from PyTorch’s torchvision.
• Method: Fast Gradient Sign Method (FGSM) to generate adversarial examples.
• Backend: Flask API for adversarial image processing.
	•	Frontend: Streamlit for an interactive UI.
