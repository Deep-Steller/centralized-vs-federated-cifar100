import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
from scipy.stats import dirichlet
import logging

# Set up logging
logging.basicConfig(filename='logs/federated_learning.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def get_transform():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

def get_test_loader():
    transform = get_transform()
    testset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
    return DataLoader(testset, batch_size=32, shuffle=False)

def get_partitioned_data(client_id, num_clients=5, non_iid=True):
    transform = get_transform()
    dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
    targets = np.array(dataset.targets)

    if non_iid:
        # Dirichlet distribution for non-IID split
        num_classes = 100
        alpha = 0.5
        client_indices = [[] for _ in range(num_clients)]
        for cls in range(num_classes):
            cls_indices = np.where(targets == cls)[0]
            np.random.shuffle(cls_indices)
            proportions = dirichlet.rvs([alpha] * num_clients, size=1)[0]
            cls_split = np.split(cls_indices, (proportions.cumsum()[:-1] * len(cls_indices)).astype(int))
            for cid, indices in enumerate(cls_split):
                client_indices[cid].extend(indices)
        indices = client_indices[client_id]
    else:
        # IID split
        data_per_client = len(dataset) // num_clients
        start = client_id * data_per_client
        end = start + data_per_client
        indices = list(range(start, end))

    # Check if the client has enough data for at least one batch
    batch_size = 32
    subset = Subset(dataset, indices)
    if len(subset) < batch_size:
        logging.warning(f"Client {client_id} has only {len(subset)} samples, which is less than the batch size {batch_size}. This may cause issues during training.")
    
    # Use drop_last=True to avoid batches with size 1
    return DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=True)