import torch
import sys
import time
import logging
import os

# Set up logging
logging.basicConfig(filename='logs/federated_learning.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def get_model_size(model):
    torch.save(model.state_dict(), "temp_model.pth")
    size = os.path.getsize("temp_model.pth")
    os.remove("temp_model.pth")
    return size

def log_communication_metrics(round_num, model, round_time):
    model_size = get_model_size(model)
    logging.info(f"Round {round_num}: Model size = {model_size} bytes, Round time = {round_time:.2f} seconds")
    with open("logs/communication_metrics.csv", "a") as f:
        f.write(f"{round_num},{model_size},{round_time}\n")