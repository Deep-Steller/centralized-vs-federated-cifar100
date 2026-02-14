import torch
import flwr as fl
from federated.client import client_fn

def simulation_main():
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=5,
        config=fl.server.ServerConfig(num_rounds=20),  
        client_resources={"num_cpus": 1, "num_gpus": 1 if torch.cuda.is_available() else 0},
    )
    return history