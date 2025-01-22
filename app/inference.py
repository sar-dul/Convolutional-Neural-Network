# inference.py
import torch
from torchvision import models
from torchvision.models import ResNet50_Weights
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.CenterCrop((224, 224)),  # Resize images
    transforms.ToTensor(),              # Convert to Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

def get_resnet_model():
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Linear(in_features=2048, out_features=2, bias=True)  # Adjusting for binary classification
    state_dict_path = 'app/ResNet_model.pth'  # Path to the saved model weights
    state_dict = torch.load(state_dict_path, map_location=torch.device('cpu'))  # Load the model onto the CPU
    model.load_state_dict(state_dict)  # Load the saved weights
    model.eval()  # Set to evaluation mode
    return model

def predict(image: Image.Image, model):
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)  # Get the predicted class (0 or 1)
    
    return "cat" if predicted.item() == 0 else "dog"
