import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.best_acc = 0
        self.counter = 0

    def step(self, acc):
        if acc > self.best_acc:
            self.best_acc = acc
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

def train_model(model, trainloader, testloader, device, max_epochs=20, lr=0.001):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_acc = []
    test_acc = []

    early_stopper = EarlyStopping(patience=5)

    for epoch in range(max_epochs):
        model.train()
        correct, total = 0, 0

        for images, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{max_epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc_epoch = correct / total
        train_acc.append(train_acc_epoch)

        # Evaluation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        test_acc_epoch = correct / total
        test_acc.append(test_acc_epoch)

        print(f"Epoch {epoch+1}: Train Acc = {train_acc_epoch:.4f}, Test Acc = {test_acc_epoch:.4f}")

        if early_stopper.step(test_acc_epoch):
            print("Early stopping triggered.")
            break

    return train_acc, test_acc

