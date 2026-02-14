from torchvision.models import resnet18
import torch.nn as nn

def get_resnet18(num_classes=100, dropout=True):
    model = resnet18(pretrained=False)
    model.fc = nn.Sequential(
        nn.Dropout(0.3) if dropout else nn.Identity(),
        nn.Linear(model.fc.in_features, num_classes)
    )
    return model
