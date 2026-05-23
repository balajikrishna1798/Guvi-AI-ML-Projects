# Emotion Detection from Uploaded Images

## Objective
Develop a comprehensive system that enables users to upload an image through a Streamlit application and accurately detect and classify the emotion present in the image using Convolutional Neural Networks (CNNs).

## System Design and Methodology

1. **User Interface Development:**
   - Designed a Streamlit web application that provides an intuitive interface for users to upload images.
   - Incorporated file type validation (only jpg, jpeg, png) and file size constraints to ensure system robustness.
   
2. **Facial Detection Implementation:**
   - Used Google's MediaPipe Face Detection for robust and efficient facial detection.
   - Extracts bounding boxes of detected faces to isolate facial features.

3. **Facial Feature Extraction:**
   - The detected faces are cropped and pre-processed (resized to 48x48 and converted to grayscale) to align with the input requirements of the emotion classification model.

4. **Emotion Classification:**
   - Implemented a custom Convolutional Neural Network (CNN) architecture using PyTorch.
   - The CNN architecture consists of multiple convolutional layers with max pooling, dropout for regularization, and fully connected layers.
   - Designed to be trained on the FER-2013 dataset, which includes 7 emotion classes: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.

## Experimental Results
(Note: To be updated post full model training)
- **Model Training:** The CNN is trained utilizing the Cross-Entropy Loss function and Adam optimizer.
- **Performance Evaluation:** Expected to measure performance in terms of Accuracy, Precision, Recall, and F1-Score on the FER-2013 test set.

## Ethical Analysis
Emotion detection technology carries significant ethical implications:
1. **Privacy Concerns:** Capturing and processing facial images requires strict adherence to privacy norms. Images uploaded to the application should not be stored persistently without explicit consent.
2. **Bias Mitigation:** Machine learning models trained on datasets like FER-2013 can exhibit bias related to age, gender, or ethnicity. It's crucial to evaluate model performance across demographic subgroups to ensure fairness and prevent discriminatory outcomes.
3. **Misinterpretation:** Emotions are complex and subjective. Relying purely on facial expressions may lead to inaccurate conclusions about a person's actual emotional state. The system's predictions should be treated as estimations rather than absolute truth.

## Conclusion
This project successfully integrates computer vision, machine learning, and web development to create an end-to-end emotion detection system. Future work could involve refining the CNN architecture, utilizing ensemble methods, and incorporating multi-modal inputs (e.g., audio) for improved accuracy.
