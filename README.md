# Centralized vs Federated Learning on CIFAR-100

## 📖 Overview

This project presents a comparative study of Centralized Learning (CL) and Federated Learning (FL) on the CIFAR-100 dataset using PyTorch and the Flower framework.  
The objective is to analyze differences in performance, convergence behavior, communication cost, and privacy trade-offs in distributed deep learning.

---

## 🎯 Objectives

- Implement centralized training using ResNet18  
- Simulate federated learning with FedAvg (5 clients)  
- Evaluate performance under non-IID distribution (Dirichlet α = 0.5)  
- Measure communication overhead per round  
- Integrate Differential Privacy using Opacus  
- Compare convergence and accuracy between CL and FL  

---

## 🛠 Tech Stack

- Python  
- PyTorch  
- Torchvision  
- Flower (Federated Learning)  
- Opacus (Differential Privacy)  
- NumPy, Pandas  
- Matplotlib  

---

## 🧠 Methodology

### Centralized Learning
- Model: ResNet18  
- Optimizer: SGD  
- Dataset: CIFAR-100  
- Training on full dataset  
- Evaluation on test set  

### Federated Learning
- Framework: Flower (FedAvg)  
- 5 simulated clients  
- Non-IID data split (Dirichlet α = 0.5)  
- 5 communication rounds  
- Server-side weight aggregation  
- Optional Differential Privacy (ε = 1.0, δ = 1e-5)  

---

## 📊 Results Summary

| Method        | Test Accuracy | Key Observation |
|--------------|--------------|----------------|
| Centralized  | ~51.7%       | Faster convergence |
| Federated    | ~30%         | Slower under non-IID setting |

Additional Findings:
- Approx. ~45MB communication per federated round  
- Non-IID distribution significantly impacts convergence  
- Differential Privacy introduces minor performance trade-off  

---

## 📂 Project Structure

```
centralized/        → Centralized training implementation
federated/          → Federated learning simulation
results/            → Plots and evaluation outputs
Report/              → Final project report
requirements.txt    → Python dependencies
README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/Deep-Steller/centralized-vs-federated-cifar100.git
cd centralized-vs-federated-cifar100
pip install -r requirements.txt
```

## 📦 Dataset

This project uses the CIFAR-100 dataset.

The dataset is automatically downloaded using torchvision when the training script is executed:

```python
datasets.CIFAR100(root="./data", download=True)
```

No manual dataset download is required.
---

## ▶️ Running the Project

### Centralized Training

```bash
python centralized/train_centralized.py
```

### Federated Learning Simulation

```bash
python federated/simulate_flower.py
```

---

## 📈 Visualizations

Generated plots include:

- Centralized training accuracy  
- Centralized vs Federated comparison  
- Communication metrics analysis  

Available in the `results/` directory.

---

## 🔒 Privacy Component

Differential Privacy is integrated using Opacus.

- Epsilon (ε) = 1.0  
- Delta (δ) = 1e-5  

This enables experimentation with privacy-utility trade-offs in federated training.

---

## 📄 Report

The detailed report is available in:

```
paper/CL_vs_FL_CIFAR100.pdf
```

---

## 🚀 Future Improvements

- Increase federated training rounds  
- Experiment with IID vs Non-IID splits  
- Hyperparameter tuning  
- Secure aggregation implementation  
- Larger-scale client simulation  

---

## 👤 Author

Pradeepa Chakkaravarthy, Logeswaran Selvapandian  
M.S. Data Science  
University of Texas at Arlington
