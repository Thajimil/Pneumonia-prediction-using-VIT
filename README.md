# Pneumonia-prediction-using-VIT
import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
import timm 
import numpy as np

# Set page config
st.set_page_config(page_title="Pneumonia Prediction", layout="centered")

# Load your trained model
@st.cache_resource
def load_model():
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=3)
    model.load_state_dict(torch.load('vit_covid_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()
class_names = ['NORMAL', 'PNEUMONIA', 'COVID-19']

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Title
st.markdown("<h1 style='text-align: center;'>Pneumonia Prediction</h1>", unsafe_allow_html=True)

# Upload section
uploaded_file = st.file_uploader("Choose a Chest X-ray Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded X-ray Image', use_column_width=True)

    # Preprocess and predict
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1).numpy().flatten()
    
    pred_idx = np.argmax(probs)
    pred_class = class_names[pred_idx]
    confidence = probs[pred_idx] * 100

    # Show prediction
    st.markdown(f"### Prediction: **{pred_class}** detected ({confidence:.2f}%)")
    st.progress(int(confidence))

    # Show breakdown
    st.subheader("Prediction Breakdown:")
    for i, prob in enumerate(probs):
        st.write(f"{class_names[i]}: {prob * 100:.2f}%")
