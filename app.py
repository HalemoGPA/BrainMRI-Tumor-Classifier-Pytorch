import os
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from src.model import MyModel, load_model
from src.utils import predict

# Load the trained model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = os.path.join("models", "model_38")
model = load_model(model_path, device)

# Define the transformation
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# map labels from int to string
label_dict = {
    0: "No Tumor",
    1: "Pituitary",
    2: "Glioma",
    3: "Meningioma",
    4: "Other",
}

# process image got from user before passing to the model
def preprocess_image(image):
    preprocessed_image = transform(image).unsqueeze(0)
    return preprocessed_image

# sample image loader
@st.cache_data
def load_sample_images(sample_images_dir):
    sample_image_files = os.listdir(sample_images_dir)
    sample_images = []
    for sample_image_file in sample_image_files:
        sample_image_path = os.path.join(sample_images_dir, sample_image_file)
        sample_image = Image.open(sample_image_path).convert("RGB")
        sample_image = sample_image.resize((150, 150))  # Resize to a fixed size
        sample_images.append((sample_image_file, sample_image))
    return sample_images

# Streamlit app
st.title("Brain Tumor Classification")


# Display sample images section
st.subheader("Sample Images")
st.write(
    "Here are some sample images. Your uploaded image should be similar to these for best results."
)

sample_images_dir = "sample"
sample_images = load_sample_images(sample_images_dir)

# Create a grid layout for sample images
num_cols = 3  # Number of columns in the grid
cols = st.columns(num_cols)

for i, (sample_image_file, sample_image) in enumerate(sample_images):
    col_idx = i % num_cols
    with cols[col_idx]:
        st.image(sample_image, caption=f"Sample {i+1}", use_column_width=True)


st.write("Upload an image below to classify it.")


# image from user
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width = 210)

    # Preprocess the image
    preprocessed_image = preprocess_image(image).to(device)
    # Make prediction
    predicted_class = predict(model, preprocessed_image, device)

    st.write(
        f"<h1 style='font-size: 48px;'>Prediction: {label_dict[predicted_class]}</h1>",
        unsafe_allow_html=True,
    )
