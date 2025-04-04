import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
from sklearn.utils import class_weight
from PIL import Image

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data transformations
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(150),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(40),
    transforms.RandomAffine(0, translate=(0.2, 0.2)),
    transforms.RandomAffine(0, shear=20),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder(root=r"D:\sdp\ll-first\dataset\training", transform=train_transforms)
test_dataset = datasets.ImageFolder(root=r"D:\sdp\ll-first\dataset\test", transform=test_transforms)

# Compute class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_dataset.targets),
                                                  y=train_dataset.targets)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load pre-trained VGG16 model
base_model = models.vgg16(pretrained=True)
for param in base_model.parameters():
    param.requires_grad = False  # Freeze the base model

# Modify the classifier part of the model
base_model.classifier = nn.Sequential(
    nn.Linear(25088, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 3),
    nn.Softmax(dim=1)
)

base_model = base_model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(base_model.parameters(), lr=0.0001)


# Training function
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=2):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(test_loader)
        val_accuracy = 100 * correct / total
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")


# Initial training
train_model(base_model, train_loader, test_loader, criterion, optimizer, num_epochs=2)

# Fine-tuning: Unfreeze the last 10 layers
for param in base_model.features[-10:].parameters():
    param.requires_grad = True

# Recompile the model with a lower learning rate
optimizer = optim.Adam(base_model.parameters(), lr=1e-5)

# Fine-tuning training
train_model(base_model, train_loader, test_loader, criterion, optimizer, num_epochs=1)

# Save the model
torch.save(base_model.state_dict(), 'my_image_classifier_model_finetuned.pth')


# Load and predict on a single image
def predict_image(image_path, model, device):
    model.eval()
    image = Image.open(image_path)
    image = test_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        class_labels = {0: 'cat', 1: 'dog', 2: 'human'}
        return class_labels[predicted.item()]


prediction = predict_image('img_1.png', base_model, device)
print(f"Prediction: {prediction}")