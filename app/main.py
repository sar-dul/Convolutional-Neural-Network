from fastapi import FastAPI, File, UploadFile
from app.inference import get_resnet_model, predict  # Import functions from inference.py
from io import BytesIO
from PIL import Image

# Initialize FastAPI app
app = FastAPI()

# Load the pre-trained ResNet model
model = get_resnet_model()

@app.post("/predict/")
async def predict_class(file: UploadFile = File(...)):
    # Load the uploaded image into PIL format
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    
    # Predict the class using the model
    predicted_class = predict(image, model)
    
    return {"predicted_class": predicted_class}