import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from model import MyModel, load_model
from utils import predict

# Load the trained model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "models\model_38"
model = load_model(model_path, device)

# Define the transformation
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


@st.cache_data
def preprocess_image(image):
    preprocessed_image = transform(image).unsqueeze(0)
    return preprocessed_image


# Streamlit app
st.title("Brain Tumor Classification")
label_dict = {
    0: "No Tumor",
    1: "Pituitary",
    2: "Glioma",
    3: "Meningioma",
    4: "Other",
}
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    preprocessed_image = preprocess_image(image).to(device)
    # Make prediction
    predicted_class = predict(model, preprocessed_image, device)

    st.write(
        f"<h1 style='font-size: 48px;'>Prediction: {label_dict[predicted_class]}</h1>",
        unsafe_allow_html=True,
    )
