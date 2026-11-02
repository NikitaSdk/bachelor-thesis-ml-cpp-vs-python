from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import time
import os
import csv

# Configure device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load training CSV files
X = pd.read_csv("mnist_train.csv").to_numpy(dtype=np.float32) / 255.0
y = pd.read_csv("mnist_train_labels.csv").to_numpy(dtype=np.int64).flatten()

# Convert to PyTorch tensors and reshape
X = torch.from_numpy(X).view(-1, 1, 28, 28)  # (N, C, H, W)
y = torch.from_numpy(y)

# Create dataset and dataloader
train_ds = torch.utils.data.TensorDataset(X, y)
loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)

# Define LeNet CNN architecture
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.classifier(x)
        return x

# Initialize network, optimizer and loss function
net = LeNet().to(device)
opt = optim.Adam(net.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training settings
num_epochs = 15
export_dir = Path("models")
export_dir.mkdir(exist_ok=True)

# Fixed metrics file
metrics_file = Path("Train_Metrics.csv")

# Create and write headers to CSV file (overwrite if exists)
with open(metrics_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Epoch', 'Train_Accuracy', 'Train_Loss'])

# Dummy input for ONNX export
dummy = torch.randn(1, 1, 28, 28).to(device)

print("Starting training...")
start_time = time.time()

for epoch in range(num_epochs):
    net.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (xb, yb) in enumerate(loader):
        xb, yb = xb.to(device), yb.to(device)
        
        opt.zero_grad()
        outputs = net(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        opt.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += yb.size(0)
        correct += predicted.eq(yb).sum().item()
    
    # Calculate epoch statistics
    epoch_loss = total_loss / (batch_idx + 1)
    epoch_acc = 100. * correct / total
    
    # Save metrics to CSV
    with open(metrics_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, f"{epoch_acc:.2f}", f"{epoch_loss:.4f}"])
    
    print(f"Epoch {epoch+1} Train Accuracy: {epoch_acc:.2f}%  Train Loss: {epoch_loss:.4f}")
    
    # Export model after each epoch
    net.eval()
    onnx_path = export_dir / f"lenet_epoch{epoch+1}.onnx"
    torch.onnx.export(
        net, dummy, onnx_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=14
    )

training_time = time.time() - start_time
final_model_size = os.path.getsize(export_dir / "lenet_epoch15.onnx") / 1024  # Convert to KB

print("\nPerformance Metrics:")
print(f"  - Training time: {training_time:.2f} seconds")
print(f"  - Model size: {final_model_size:.2f} KB")
print(f"  - Training metrics saved to: {metrics_file}")

print("\nTraining completed!")
