import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from sklearn.model_selection import KFold
import warnings

warnings.filterwarnings("ignore")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper-parameters
num_epochs = 5
batch_size = 10
learning_rate = 0.001

# We transform the Train Data to Tensors of normalized range [-1, 1]
transform = transforms.Normalize((0.5), (0.5))

# Loading the Data and adding 1 dimension at the channel position as its a Dataset of Grayscale images:
data = torch.load("./strange_symbols/training_data.pt").unsqueeze(1).type(torch.float32)
classes = np.array(torch.load("./strange_symbols/training_labels.pt"))

data = transform(data)


# Custom Dataset Class
class SymbolDataset(Dataset):

    def __init__(self, data, labels):
        self.data = data
        self.classes = torch.from_numpy(labels)
        self.n_samples = self.data.size()[0]

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.data[index], self.classes[index]

    # returns the size
    def __len__(self):
        return self.n_samples


# Data Splitting [80-20] Train-Validation set split
split = int(data.size()[0] * 0.8)
train_dataset = SymbolDataset(data[:split, :, :, :], classes[:split])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Custom CNN Model [Conv -> Pool -> Conv -> FCL1 -> FCL2 -> Prediction]
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 15)

    def forward(self, x):
        # -> 10, 1, 28, 28
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 32, 14, 14
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 64, 7, 7
        x = x.view(-1, 64 * 7 * 7)  # -> n, 3136
        x = F.relu(self.fc1(x))  # -> n, 120
        x = self.fc2(x)  # -> n, 15
        return x


model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    model.train()
    for step, (images, labels) in enumerate(train_loader):
        # origin shape: [10, 1, 28, 28] = 10, 1, 784
        # input_layer: 1 input channels, 32 output channels, 3 kernel size
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward Propagation and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 400 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{n_total_steps}], Loss: {loss.item():.4f}"
            )

# Saving the model
print("...Finished Training\n\n")
PATH = "./cnn.pth"
torch.save(model.state_dict(), PATH)


# Set random seed for reproducibility
torch.manual_seed(42)

# 5 K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold = 0
val_x = data[split:, :, :, :]
val_y = classes[split:]
for train_idx, test_idx in kf.split(val_x):
    fold += 1
    print(f"Fold #{fold}")
    x_train, x_test = val_x[train_idx, :, :, :], val_x[test_idx, :, :, :]
    y_train, y_test = val_y[train_idx], val_y[test_idx]
    y_test = torch.from_numpy(y_test)
    val_dataset = SymbolDataset(x_train, y_train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    for _ in range(num_epochs):
        model.train()
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            val_output = model(x_test)
            val_loss = criterion(val_output, y_test)

    print(f"Validation Loss: {val_loss.item()}")

# Final evaluation
model.eval()
with torch.no_grad():
    oos_pred = model(x_test)

    n_correct, n_samples = 0, 0

    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

# Total RMSE Score
score = torch.sqrt(criterion(oos_pred, y_test)).item()
print(f"Fold score (RMSE): {score}")
