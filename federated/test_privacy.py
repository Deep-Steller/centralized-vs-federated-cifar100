import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision import transforms
from privacy_utils import apply_differential_privacy

# Create a simple model and data loader for testing
model = nn.Linear(3072, 100)  # Dummy model for CIFAR-100 (flattened 32x32x3 images)
optimizer = optim.Adam(model.parameters(), lr=0.005)
transform = transforms.Compose([transforms.ToTensor()])
dataset = CIFAR100(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Apply differential privacy
model, optimizer, dataloader = apply_differential_privacy(model, optimizer, dataloader, epsilon=1.0, delta=1e-5)

print("Differential privacy applied successfully!")