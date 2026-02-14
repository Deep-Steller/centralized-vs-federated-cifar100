from comms_utils import log_communication_metrics
from model import get_resnet18

# Create a model instance
model = get_resnet18()

# Log communication metrics for a test round
log_communication_metrics(round_num=0, model=model, round_time=1.0)

print("Communication metrics logged successfully!")