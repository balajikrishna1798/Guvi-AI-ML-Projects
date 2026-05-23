import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import EmotionCNN
import os

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # FER2013 data transforms
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    print("Preparing dataset...")
    try:
        train_dataset = datasets.FER2013(root='./data', split='train', transform=transform, download=True)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    except Exception as e:
        print("Note: Could not load FER2013 dataset automatically. Please ensure it's downloaded.")
        return
        
    model = EmotionCNN(num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 10
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
        
    torch.save(model.state_dict(), 'emotion_model.pth')
    print("Training complete. Model saved to emotion_model.pth")

if __name__ == '__main__':
    train()
