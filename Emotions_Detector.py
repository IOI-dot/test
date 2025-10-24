import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
from sklearn.utils.class_weight import compute_class_weight
Data_Folder= r"C:\Users\Omar\Downloads\Emotions"
Batch_Size = 16
Image_Size = 128
EPOCHS = 30
LR = 0.001
train_transform = transforms.Compose([transforms.Resize((Image_Size,Image_Size)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(10),
                                      transforms.ColorJitter(brightness=0.2,contrast=0.2),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5, 0.5, 0.5],[0.5,0.5,0.5])])
valid_transform = transforms.Compose([transforms.Resize((Image_Size,Image_Size)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5, 0.5, 0.5],[0.5,0.5,0.5])])
train_dataset = datasets.ImageFolder(os.path.join(Data_Folder,"train"), transform=train_transform)
valid_dataset = datasets.ImageFolder(os.path.join(Data_Folder,"test"), transform=valid_transform)
train_loader = DataLoader(train_dataset, batch_size=Batch_Size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=Batch_Size, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Net(nn.Module):
    def __init__(self, num_classes,img_size=Image_Size):
        super().__init__()
        self.feature_extractor = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(32),
                                               nn.ELU(),nn.MaxPool2d(kernel_size=2, stride=2),
                                               nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1,
                                                         padding=1),nn.BatchNorm2d(64),
                                               nn.ELU(), nn.MaxPool2d(kernel_size=2, stride=2),
                                               nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1,
                                                         padding=1),nn.BatchNorm2d(128),
                                               nn.ELU(), nn.MaxPool2d(kernel_size=2, stride=2),
                                               nn.Flatten()
                                               )
        img_size=img_size//8
        self.classifier = nn.Sequential(nn.Linear(in_features=128 * img_size * img_size, out_features=1024),
                                        nn.ReLU(),
                                        nn.Linear(in_features=1024,out_features= num_classes))

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

model = Net(num_classes=len(train_dataset.classes), img_size=Image_Size).to(device)
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_dataset.targets),
    y=train_dataset.targets
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

for epoch in range(EPOCHS):
    model.train()
    running_corrects = 0
    for i, (inputs, labels) in enumerate(train_loader):
        if i % 10 == 0:
            print(f"Batch {i + 1}/{len(train_loader)}")
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
    train_acc = running_corrects.double() / len(train_dataset)

    model.eval()
    running_corrects = 0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
        val_acc = running_corrects.double() / len(valid_dataset)
        print(f"Epoch {epoch + 1}/{EPOCHS} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")

    scheduler.step(val_acc)


torch.save(model.state_dict(), "emotions_cnn.pth")

