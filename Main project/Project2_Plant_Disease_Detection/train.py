import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import CustomPlantDiseaseCNN, get_pretrained_model
import os

# Set dataset path and parameters
DATA_DIR = './dataset' # Placeholder path, user should download dataset here
NUM_CLASSES = 38 # Assuming standard New Plant Diseases Dataset (38 classes)
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

def train_and_evaluate(model_name, model, train_loader, val_loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"\n--- Training {model_name} ---")
    
    best_acc = 0.0
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {epoch_loss:.4f} - Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{model_name}_best.pth")
            
    print(f"Best Validation Accuracy for {model_name}: {best_acc:.2f}%")
    return best_acc

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not os.path.exists(DATA_DIR):
        print(f"Dataset directory '{DATA_DIR}' not found. Please download the dataset and place it here.")
        return
        
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Assuming standard folder structure: dataset/train and dataset/valid
    try:
        train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transform)
        val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'valid'), transform=val_transform)
    except FileNotFoundError:
        print("Please structure dataset as dataset/train and dataset/valid with subfolders for each class.")
        return
        
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Models to train
    models_to_train = {
        'CustomCNN': CustomPlantDiseaseCNN(num_classes=NUM_CLASSES),
        'ResNet18': get_pretrained_model('resnet18', NUM_CLASSES),
        'VGG16': get_pretrained_model('vgg16', NUM_CLASSES),
        'MobileNetV2': get_pretrained_model('mobilenet_v2', NUM_CLASSES)
    }
    
    results = {}
    
    for name, model in models_to_train.items():
        model = model.to(device)
        best_acc = train_and_evaluate(name, model, train_loader, val_loader, device)
        results[name] = best_acc
        
    print("\n--- Final Results ---")
    for name, acc in results.items():
        print(f"{name}: {acc:.2f}%")

if __name__ == '__main__':
    main()
