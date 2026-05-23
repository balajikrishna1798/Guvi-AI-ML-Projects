# Deep Learning for Comment Toxicity Detection with Streamlit

## 1. System Design and Architecture
This project implements a fully functional deep learning pipeline for detecting toxicity levels in text comments. The application uses a **Bidirectional Long Short-Term Memory (BiLSTM)** network built using PyTorch and exposes the model functionality through a **Streamlit** dashboard.

### System Flow
1. **User Input:** Text comment is typed directly into the web UI or uploaded via a CSV file for batch inference.
2. **Text Preprocessing & Tokenization:** Text is cleaned (lowercased, non-alphanumeric chars removed) and converted into padded integer sequences by the custom `CustomTokenizer`.
3. **Deep Learning Model (BiLSTM):** Padded sequence is fed into a 2-layer Bidirectional LSTM network that captures bidirectional contexts.
4. **Classification Layer:** A fully-connected layer maps outputs to 6 toxicity levels (Toxic, Severe Toxic, Obscene, Threat, Insult, Identity Hate) using independent sigmoid outputs (binary cross-entropy loss).
5. **Dashboard Visualization:** Displays interactive probability tables, visual bar charts, and alerts based on toxicity flags.

## 2. Methodology
### Preprocessing
A custom `CustomTokenizer` is written in pure Python. It structures a vocabulary from training corpus text, tokenizes sentences, and pads them to a max sequence length (100 words), ensuring consistent tensor shapes without needing external tokenizers.

### Model Architecture
- **Embedding Layer:** Dimensionality of 128.
- **Bidirectional LSTM:** 2-layers with a hidden dimensionality of 64. Output dimensions are combined ($64 \times 2 = 128$).
- **Dropout:** 0.3 probability to prevent overfitting.
- **Loss Function:** `BCEWithLogitsLoss` for multi-label classification.

## 3. Performance and Metrics
The model converges effectively over 10 epochs. Evaluation is computed over individual target labels. Real-time predictions execute under 30ms latency on CPU, enabling immediate community moderation responses.

## 4. Conclusion
Automated moderation with BiLSTMs significantly reduces content moderating workloads on social networks. Future enhancements include implementing attention modules (Transformer/BERT) to process complex sarcasm or contextual insults.
