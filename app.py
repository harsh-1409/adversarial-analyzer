import os
from flask import Flask, request, jsonify
from torchvision import models, transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.autograd import Variable
import torch
import numpy as np
from PIL import Image
import json

import logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
upload_folder = 'uploads'
os.makedirs(upload_folder, exist_ok=True)
app.config['UPLOAD_FOLDER'] = upload_folder

model = models.resnet50(weights = ResNet50_Weights.DEFAULT)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def generate_adversarial_example(model, img_tensor, epsilon=0.5):
    img_tensor.requires_grad = True

    #Forward pass
    output = model(img_tensor)
    _, target_class = output.max(1)

    #Loss
    loss = torch.nn.CrossEntropyLoss()(output, torch.tensor([target_class]))

    #Backward pass
    model.zero_grad()
    loss.backward()

    #FGSM
    adv_img_tensor = img_tensor + epsilon * img_tensor.grad.sign()
    adv_img_tensor = torch.clamp(adv_img_tensor, 0, 1)

    return adv_img_tensor

@app.route('/upload', methods=['POST'])

def upload_file():
    
    try:
        app.logger.info("File received and processing started.")
        if 'file' not in request.files:
            return jsonify({'error': 'No file found'}), 400
        
        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        
        #Save the filepath
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        #process the image
        img = Image.open(file_path).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0) #Unsqueeze to add batch dimension
        
        epsilon = float(request.args.get('epsilon', 0.5))

        adv_img_tensor = generate_adversarial_example(model, img_tensor, epsilon)
        
        #Convert tensors to numpy arrays
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

        return jsonify({
            "original_prediction": {
                "class": int(original_class),
                "label" : original_label,
                "confidence": float(torch.softmax(output, dim=1).max().item())
            },
            "adversarial_prediction": {
                "class": int(adv_class),
                "label" : adv_label,
                "confidence": float(torch.softmax(adv_output, dim=1).max().item())
            }
        })
    except Exception as e:
        app.logger.error(f"Error processing the image: {e}")
        return jsonify({'error': 'Error processing the image'}), 500

if __name__ == '__main__':
    app.run(debug=True)