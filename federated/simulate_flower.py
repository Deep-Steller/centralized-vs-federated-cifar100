import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import multiprocessing
import time
import logging
import pandas as pd
import matplotlib.pyplot as plt
from federated.server import main as server_main
from federated.run_simulation import simulation_main

# Set up logging
logging.basicConfig(filename='logs/federated_learning.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    logging.info("Starting federated learning simulation")

    # Start server in a separate process
    server_process = multiprocessing.Process(target=server_main)
    server_process.start()

    # Wait for server to initialize
    time.sleep(5)

    # Run simulation
    try:
        history = simulation_main()

        # Process and save metrics
        metrics = []
        for rnd, loss in history.losses_distributed:
            if loss is not None:
                metrics.append((rnd, loss))
        for rnd, acc in history.metrics_distributed.get("accuracy", []):
            if acc is not None:
                metrics.append((rnd, None, acc))

        # Save federated metrics to CSV
        df_fed = pd.DataFrame(metrics, columns=["round", "loss", "accuracy"])
        df_fed.to_csv("logs/federated_metrics.csv", index=False)
        logging.info("Federated metrics saved to logs/federated_metrics.csv")

        # Load centralized metrics for comparison
        df_cent = pd.read_csv("logs/centralized_metrics.csv")

        # Plot centralized vs federated
        plt.figure(figsize=(10, 5))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(df_cent["epoch"], df_cent["train_accuracy"], label="Centralized Train", marker="o")
        plt.plot(df_cent["epoch"], df_cent["val_accuracy"], label="Centralized Val", marker="o")
        plt.plot(df_fed["round"], df_fed["accuracy"], label="Federated", marker="o", color="green")
        plt.xlabel("Epoch/Round")
        plt.ylabel("Accuracy")
        plt.title("Centralized vs Federated Accuracy")
        plt.legend()
        plt.grid()

        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(df_fed["round"], df_fed["loss"], label="Federated Loss", marker="o", color="red")
        plt.xlabel("Round")
        plt.ylabel("Loss")
        plt.title("Federated Loss")
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.savefig("results/centralized_vs_federated.png")
        plt.close()
        logging.info("Plot saved to results/centralized_vs_federated.png")

        # Plot federated accuracy vs round separately
        plt.figure(figsize=(6, 4))
        plt.plot(df_fed["round"], df_fed["accuracy"], label="Federated Accuracy", marker="o", color="green")
        plt.xlabel("Round")
        plt.ylabel("Accuracy")
        plt.title("Federated Accuracy vs Round")
        plt.legend()
        plt.grid()
        plt.savefig("results/accuracy_vs_round.png")
        plt.close()
        logging.info("Federated accuracy plot saved to results/accuracy_vs_round.png")

    except Exception as e:
        logging.error(f"Simulation error: {str(e)}")

    finally:
        server_process.terminate()
        server_process.join()
        logging.info("Federated learning simulation complete")