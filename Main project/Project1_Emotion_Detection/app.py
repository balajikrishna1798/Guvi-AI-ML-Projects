import streamlit as st
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import mediapipe as mp
import os
from model import EmotionCNN

# Emotion mapping for FER-2013
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

@st.cache_resource
def load_model():
    model = EmotionCNN(num_classes=7)
    model_path = 'emotion_model.pth'
    if os.path.exists(model_path):
        # Load with map_location to ensure it works on CPU if trained on GPU
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    return None

def process_image(image_bytes, model):
    # Convert bytes to numpy array then to OpenCV format
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    results = face_detection.process(img_rgb)
    
    if not results.detections:
        return img_rgb, None
        
    for detection in results.detections:
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, _ = img.shape
        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
        
        # Ensure bounding box is within image bounds
        x, y = max(0, x), max(0, y)
        x_end, y_end = min(iw, x+w), min(ih, y+h)
        
        face_img = img_rgb[y:y_end, x:x_end]
        if face_img.size == 0:
            continue
            
        if model is not None:
            # Prepare image for PyTorch model
            pil_img = Image.fromarray(face_img).convert('L')
            transform = transforms.Compose([
                transforms.Resize((48, 48)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            input_tensor = transform(pil_img).unsqueeze(0)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                _, predicted = torch.max(outputs, 1)
                emotion = EMOTIONS[predicted.item()]
                confidence = torch.softmax(outputs, dim=1)[0][predicted.item()].item()
        else:
            emotion = "Model not trained"
            confidence = 0.0
            
        # Draw bounding box and emotion text
        cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img_rgb, f"{emotion} ({confidence:.2f})", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
    return img_rgb, results.detections

def main():
    st.set_page_config(page_title="Emotion Detection App", layout="wide")
    st.title("Emotion Detection from Uploaded Images")
    
    st.write("Upload an image to detect faces and classify their emotions.")
    
    model = load_model()
    if model is None:
        st.warning("⚠️ Pre-trained model ('emotion_model.pth') not found. " 
                   "Please run `train.py` first to train the CNN model. "
                   "Face detection will still work.")
                   
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Check size (e.g., limit to 5MB)
        if uploaded_file.size > 5 * 1024 * 1024:
            st.error("File is too large. Please upload an image smaller than 5MB.")
            return
            
        image_bytes = uploaded_file.read()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Uploaded Image")
            st.image(image_bytes, use_column_width=True)
            
        with col2:
            st.subheader("Processed Image (Detection & Emotion)")
            with st.spinner("Processing..."):
                processed_img, detections = process_image(image_bytes, model)
                st.image(processed_img, use_column_width=True)
                
                if detections:
                    st.success(f"Detected {len(detections)} face(s).")
                else:
                    st.error("No faces detected in the image.")

if __name__ == "__main__":
    main()
