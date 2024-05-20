import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from model import MyModel, load_model
from utils import predict

# Load the trained model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = 'saved_model/model_38'
model = load_model(model_path, device)

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Streamlit app
st.title("Brain Tumor Classification")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    image = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    predicted_class = predict(model, image, device)
    
    label_dict = {0: "No Tumor", 1: "Glioma", 2: "Meningioma", 3: "Pituitary", 4: "Other"}
    st.write(f"Prediction: {label_dict[predicted_class]}")
