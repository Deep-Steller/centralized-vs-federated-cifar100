import torch
from opacus import PrivacyEngine
import logging

# Set up logging
logging.basicConfig(filename='logs/federated_learning.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def apply_differential_privacy(model, optimizer, trainloader, epsilon=1.0, delta=1e-5):
    privacy_engine = PrivacyEngine()
    model, optimizer, trainloader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=trainloader,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        poisson_sampling=False,
    )
    logging.info(f"Differential privacy applied with epsilon={epsilon}, delta={delta}")
    # Log privacy metrics to a CSV file
    with open("logs/privacy_metrics.csv", "a") as f:
        f.write(f"epsilon={epsilon},delta={delta}\n")
    return model, optimizer, trainloader