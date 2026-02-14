import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import pandas as pd
import matplotlib.pyplot as plt
from centralized.cnn_model import get_resnet18
from centralized.dataset import get_cifar100
from centralized.train_utils import train_model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_resnet18(dropout=True)

    trainloader, testloader = get_cifar100(batch_size=64)
    train_acc, test_acc = train_model(model, trainloader, testloader, device)

    # Save model
    torch.save(model.state_dict(), "models/centralized_model.pth")

    # Save logs
    df = pd.DataFrame({
        "epoch": list(range(1, len(train_acc)+1)),
        "train_accuracy": train_acc,
        "test_accuracy": test_acc
    })
    df.to_csv("logs/centralized_metrics.csv", index=False)

    # Plot
    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(test_acc, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("ResNet18 on CIFAR-100")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/centralized.png")
    plt.show()

if __name__ == "__main__":
    main()

