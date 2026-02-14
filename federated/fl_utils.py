import torch
import torch.nn as nn
import torch.optim as optim

def train(model, trainloader, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  
    total_loss, correct, total = 0.0, 0, 0

    for _ in range(3):  # 3 local epochs
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        scheduler.step()

    avg_loss = total_loss / (len(trainloader) * 3)  # Average over all batches and epochs
    accuracy = correct / total
    return avg_loss, accuracy

def test(model, testloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            _, pred = torch.max(model(x), 1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total