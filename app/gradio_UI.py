import gradio as gr
import requests
from PIL import Image
import io

# FastAPI server URL
API_URL = "http://127.0.0.1:8000/predict/"

# Define the Gradio interface function
def classify_image(image: Image.Image):
    # Convert image to bytes to send to FastAPI
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()
    
    # Make a request to FastAPI backend
    response = requests.post(API_URL, files={"file": img_byte_arr})
    
    if response.status_code == 200:
        return response.json()["predicted_class"]
    else:
        return "Error in prediction"

# Create the Gradio interface with enhanced UI
gr_interface = gr.Interface(
    fn=classify_image,                # Function to classify the image
    inputs=gr.Image(type="pil", label="Upload Image"),  # Image input with a label
    outputs=gr.Textbox(label="Predicted Class", lines=2, placeholder="Prediction will appear here"),  # Text output with label and placeholder
    live=True,                         # Enables live updates
    title="Image Classifier",         # Title of the interface
    description="Upload an image of a cat or dog to predict the class.",  # Description of the app
    theme="huggingface",               # Use the huggingface theme instead of compact
)

if __name__ == "__main__":
    # Launch Gradio interface with additional settings
    gr_interface.launch(share=True, server_name="0.0.0.0", server_port=7860, inbrowser=True)
