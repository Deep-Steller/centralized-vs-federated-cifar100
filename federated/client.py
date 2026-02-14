import flwr as fl
import torch
import torch.nn as nn
from federated.model import get_resnet18
from federated.dataset import get_partitioned_data, get_test_loader
from federated.fl_utils import train, test

class CIFARClient(fl.client.NumPyClient):
    def __init__(self, cid, total_clients=5):
        self.cid = int(cid)
        self.model = get_resnet18()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.trainloader = get_partitioned_data(self.cid, total_clients, non_iid=True)
        self.testloader = get_test_loader()

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v, device=self.device) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        # Train and compute metrics
        total_loss, accuracy = train(self.model, self.trainloader, self.device)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {
            "loss": total_loss,
            "accuracy": accuracy
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        acc = test(self.model, self.testloader, self.device)
        return float(1.0 - acc), len(self.testloader.dataset), {"accuracy": float(acc)}

def client_fn(cid: str):
    return CIFARClient(cid, total_clients=5).to_client()