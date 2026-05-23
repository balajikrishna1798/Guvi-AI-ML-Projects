# Comment Toxicity Detection App

This application analyzes online comments using a Deep Learning **Bidirectional LSTM** model to detect toxic content, helping community administrators and content moderators flag harassment, hate speech, and obscenity.

## Features
- **Real-Time Analysis:** Enter individual comments in the Streamlit interface for immediate, multi-label toxicity checks.
- **Multi-Label Classifications:** Outputs probabilities for:
  - Toxic
  - Severe Toxic
  - Obscene
  - Threat
  - Insult
  - Identity Hate
- **Bulk Inference:** Upload a `.csv` file containing comments (with a `comment_text` column) to run batch predictions and download the output.

## Installation and Setup

### Prerequisites
- Python 3.8+

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
Run the training script to train the model on sample data and create the necessary assets:
```bash
python train.py
```
This trains the Bidirectional LSTM model and saves:
- `toxicity_model.pth` (Model weights)
- `tokenizer.json` (Custom tokenizer vocabulary config)

### 3. Launch the Application
Run the Streamlit application:
```bash
streamlit run app.py
```
This will open the web interface in your default browser automatically.

## Project Structure
- `app.py`: Streamlit-based web application.
- `model.py`: Model architecture and tokenizer classes.
- `train.py`: Data pre-processing, training loop, and model export logic.
- `requirements.txt`: Python package requirements.
- `Report.md`: In-depth project report outlining design choices.
- `README.md`: User guide and execution instructions.
