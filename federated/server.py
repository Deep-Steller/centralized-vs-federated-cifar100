import flwr as fl
import numpy as np

def get_strategy():
    def fit_metrics_aggregation_fn(fit_metrics):
        # Aggregate accuracy from clients
        accuracies = [metrics["accuracy"] for _, metrics in fit_metrics]
        return {"accuracy": np.mean(accuracies)}

    def evaluate_metrics_aggregation_fn(eval_metrics):
        # Aggregate accuracy from clients
        accuracies = [metrics["accuracy"] for _, metrics in eval_metrics]
        return {"accuracy": np.mean(accuracies)}

    return fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=5,
        min_evaluate_clients=5,
        min_available_clients=5,
        on_fit_config_fn=lambda rnd: {"rnd": rnd},
        on_evaluate_config_fn=lambda rnd: {"rnd": rnd},
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    )

def main():
    strategy = get_strategy()
    fl.server.start_server(
        server_address="0.0.0.0:8085",
        config=fl.server.ServerConfig(num_rounds=20),  
        strategy=strategy,
    )

if __name__ == "__main__":
    main()