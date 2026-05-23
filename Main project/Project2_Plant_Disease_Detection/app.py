import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import json
from model import CustomPlantDiseaseCNN

# Attempt to load class labels if available, otherwise use generic labels
# For a full project, class_indices.json should be created during training.
try:
    with open('class_indices.json', 'r') as f:
        CLASS_NAMES = json.load(f)
except FileNotFoundError:
    CLASS_NAMES = {str(i): f"Disease Class {i}" for i in range(38)}

NUM_CLASSES = len(CLASS_NAMES)

@st.cache_resource
def load_model():
    model = CustomPlantDiseaseCNN(num_classes=NUM_CLASSES)
    model_path = 'CustomCNN_best.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    return None

def predict(image, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    class_idx = str(predicted.item())
    class_name = CLASS_NAMES.get(class_idx, f"Class {class_idx}")
    
    return class_name, confidence.item()

def main():
    st.set_page_config(page_title="Plant Disease Detection", layout="centered")
    
    st.title("🌿 Plant Disease Detection from Leaves")
    st.write("Upload an image of a plant leaf to identify potential diseases.")
    
    model = load_model()
    
    if model is None:
        st.warning("⚠️ Trained model 'CustomCNN_best.pth' not found. Please run the training script first. The app will not be able to predict.")
        
    uploaded_file = st.file_uploader("Choose an image of a leaf...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            if model is not None:
                with st.spinner('Analyzing...'):
                    class_name, confidence = predict(image, model)
                    
                st.success(f"**Prediction:** {class_name}")
                st.info(f"**Confidence:** {confidence:.2%}")
                
                # Feedback logic (stub)
                st.write("---")
                st.write("Was this prediction helpful?")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Yes"):
                        st.write("Thank you for your feedback!")
                with col2:
                    if st.button("No"):
                        st.write("We will use this feedback to improve the model.")
        except Exception as e:
            st.error(f"Error processing the image. Please ensure it's a valid image file. Details: {e}")

if __name__ == "__main__":
    main()
