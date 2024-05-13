import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn

# Dataset class
class DHCDataset(Dataset):
    """ Devnagari Handwritten Character Dataset class """
    def __init__(self, npz_file, train=True):
        """
        Args:
            npz_file (string): Path to the NPZ file containing the DHCD
        """
        self.__dataset_npz = np.load(npz_file)
        self.train = train
        self.image_train = self.__dataset_npz['arr_0']
        self.label_train = self.__dataset_npz['arr_1']
        self.image_test = self.__dataset_npz['arr_2']
        self.label_test = self.__dataset_npz['arr_3']

    def __len__(self):
        """
        Returns dataset size
        """
        if self.train:
            return len(self.image_train)
        else:
            return len(self.image_test)

    def __getitem__(self, idx):
        """
        Returns indexed item
        """
        if self.train:
            img, label = self.image_train[idx, ...], self.label_train[idx]
        else:
            img, label = self.image_test[idx, ...], self.label_test[idx]
        return img, label

    def __repr__(self):
        repr_str = 'Devnagari Handwritten Character Dataset \n'
        repr_str += 'Training set contains {} images\n'.format(len(self.image_train))
        repr_str += 'Testing set contains {} images\n'.format(len(self.image_test))
        return repr_str

# Data augmentation
data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(degrees=10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
}

# Data loaders
train_dataset = DHCDataset('/OCR/DataSet/dataset_DCHD.npz', train=True)
val_dataset = DHCDataset('/OCR/DataSet/dataset_DCHD.npz', train=False)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ResNet50 model
class ResNet50Model(nn.Module):
    def __init__(self, num_classes=46):
        super(ResNet50Model, self).__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

model = ResNet50Model()

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Evaluate on validation set
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Accuracy: {accuracy:.2f}%')