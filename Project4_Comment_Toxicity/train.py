import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from model import ToxicityLSTM, CustomTokenizer
import os

CLASSES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def generate_mock_data():
    """Generates a small dataset representing comment toxicity classes."""
    data = [
        ("I love this project, excellent work!", 0, 0, 0, 0, 0, 0),
        ("This is the worst comment ever, I hate you.", 1, 0, 0, 0, 1, 0),
        ("Go to hell, you stupid idiot.", 1, 1, 1, 0, 1, 1),
        ("The weather is nice today.", 0, 0, 0, 0, 0, 0),
        ("I am going to kill you, stay away from me.", 1, 1, 0, 1, 0, 0),
        ("This platform is extremely useful.", 0, 0, 0, 0, 0, 0),
        ("You suck so bad, please stop coding.", 1, 0, 1, 0, 1, 0),
        ("Excellent documentation, very clear.", 0, 0, 0, 0, 0, 0),
        ("Stupid monkey, go back to your country.", 1, 1, 0, 0, 1, 1),
        ("Thank you for your feedback.", 0, 0, 0, 0, 0, 0)
    ]
    # Multiply to simulate a larger dataset
    data = data * 50
    df = pd.DataFrame(data, columns=['comment_text'] + CLASSES)
    return df

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Load dataset
    print("Loading data...")
    # In a real environment, load standard Kaggle train.csv:
    # df = pd.read_csv('train.csv')
    df = generate_mock_data()
    
    # Tokenize
    tokenizer = CustomTokenizer(num_words=5000, max_len=100)
    tokenizer.fit_on_texts(df['comment_text'].values)
    
    sequences = tokenizer.texts_to_sequences(df['comment_text'].values)
    padded_sequences = tokenizer.pad_sequences(sequences)
    
    # Convert to Tensors
    X = torch.tensor(padded_sequences, dtype=torch.long)
    y = torch.tensor(df[CLASSES].values, dtype=torch.float)
    
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Init model
    model = ToxicityLSTM(vocab_size=5000, embedding_dim=128, hidden_dim=64, num_classes=len(CLASSES)).to(device)
    
    criterion = nn.BCEWithLogitsLoss() # Good for multi-label classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 10
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)
            
        epoch_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}")
        
    # Save Model and Tokenizer
    torch.save(model.state_dict(), 'toxicity_model.pth')
    with open('tokenizer.json', 'w') as f:
        f.write(tokenizer.to_json())
        
    print("Training finished. Saved model and tokenizer.")

if __name__ == '__main__':
    train()
