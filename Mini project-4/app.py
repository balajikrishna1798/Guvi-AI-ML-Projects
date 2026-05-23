import streamlit as st
import torch
import pandas as pd
import numpy as np
import os
import json
from model import ToxicityLSTM, CustomTokenizer

CLASSES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

@st.cache_resource
def load_assets():
    model_path = 'toxicity_model.pth'
    tokenizer_path = 'tokenizer.json'
    
    if os.path.exists(model_path) and os.path.exists(tokenizer_path):
        with open(tokenizer_path, 'r') as f:
            tokenizer_json = f.read()
        tokenizer = CustomTokenizer.from_json(tokenizer_json)
        
        # Vocab size needs to match training (5000)
        model = ToxicityLSTM(vocab_size=5000, embedding_dim=128, hidden_dim=64, num_classes=len(CLASSES))
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model, tokenizer
        
    return None, None

def predict_toxicity(text, model, tokenizer):
    seqs = tokenizer.texts_to_sequences([text])
    padded = tokenizer.pad_sequences(seqs)
    
    inputs = torch.tensor(padded, dtype=torch.long)
    with torch.no_grad():
        outputs = model(inputs)
        # Apply sigmoid to output logits to get probabilities
        probabilities = torch.sigmoid(outputs).squeeze().numpy()
        
    return {CLASSES[i]: float(probabilities[i]) for i in range(len(CLASSES))}

def main():
    st.set_page_config(page_title="Toxicity Detection App", layout="wide")
    
    st.title("🛡️ Deep Learning Comment Toxicity Detector")
    st.write("Analyze online comments in real-time to detect harassment, hate speech, and toxicity.")
    
    model, tokenizer = load_assets()
    
    if model is None:
        st.warning("⚠️ Trained model (`toxicity_model.pth`) or tokenizer (`tokenizer.json`) not found. "
                   "Please run `python train.py` first to train the Bidirectional LSTM.")
                   
    # Selection mode
    mode = st.radio("Choose Input Mode", ["Single Comment Analysis", "Bulk Analysis (CSV Upload)"])
    
    if mode == "Single Comment Analysis":
        st.subheader("Single Comment Analysis")
        user_input = st.text_area("Enter comment to analyze:", placeholder="Type your comment here...")
        
        if st.button("Analyze") and user_input:
            if model is not None:
                with st.spinner("Analyzing text..."):
                    results = predict_toxicity(user_input, model, tokenizer)
                    
                st.subheader("Results:")
                
                # Check if any label is > 0.5
                is_toxic = any(prob >= 0.5 for prob in results.values())
                if is_toxic:
                    st.error("🚨 Warning: Toxic content detected!")
                else:
                    st.success("✅ Clean: No toxic content detected.")
                    
                # Render predictions
                col1, col2 = st.columns(2)
                with col1:
                    df_res = pd.DataFrame(results.items(), columns=["Toxicity Class", "Probability"])
                    st.dataframe(df_res.style.format({"Probability": "{:.2%}"}))
                with col2:
                    st.bar_chart(df_res.set_index("Toxicity Class"))
            else:
                st.error("Model is not initialized.")
                
    elif mode == "Bulk Analysis (CSV Upload)":
        st.subheader("Bulk Analysis")
        st.write("Upload a CSV file containing comments to run prediction on multiple inputs. The CSV must have a `comment_text` column.")
        
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        
        if uploaded_file is not None and model is not None:
            df = pd.read_csv(uploaded_file)
            if 'comment_text' in df.columns:
                with st.spinner("Processing batch predictions..."):
                    predictions = []
                    for text in df['comment_text'].values:
                        results = predict_toxicity(str(text), model, tokenizer)
                        predictions.append(results)
                        
                # Add columns to dataframe
                for c in CLASSES:
                    df[c] = [round(pred[c], 4) for pred in predictions]
                    
                st.success("Batch processing complete!")
                st.dataframe(df, use_container_width=True)
                
                # Download results button
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name="toxicity_predictions.csv",
                    mime="text/csv"
                )
            else:
                st.error("CSV file must contain a 'comment_text' column.")

if __name__ == "__main__":
    main()
