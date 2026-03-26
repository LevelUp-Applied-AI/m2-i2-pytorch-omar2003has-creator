import torch
import torch.nn as nn
import numpy as np
import itertools
import json
import time
import pandas as pd
import matplotlib

import matplotlib.pyplot as plt
matplotlib.use('Agg') 
import matplotlib.pyplot as plt



learning_rates = [0.1, 0.01, 0.001]
hidden_sizes = [16, 32, 64, 128, 256]
epochs_list = [100, 200]  

all_configs = list(itertools.product(learning_rates, hidden_sizes, epochs_list))

class HousingModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x

torch.manual_seed(42)
df = pd.read_csv('predictions.csv')
X = df.drop('actual', axis=1).values
y = df['actual'].values

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

indices = torch.randperm(len(X_tensor))
split = int(0.8 * len(X_tensor))
X_train, X_test = X_tensor[indices[:split]], X_tensor[indices[split:]]
y_train, y_test = y_tensor[indices[:split]], y_tensor[indices[split:]]

def r2_score(y_true, y_pred):
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return (1 - ss_res / ss_tot).item()

results = []
print(f"Starting {len(all_configs)} experiments...")

for i, (lr, hidden, epochs) in enumerate(all_configs):
    model = HousingModel(X_train.shape[1], hidden)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    
    train_loss = loss.item()
    duration = time.time() - start_time
    
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test)
        test_loss = criterion(test_preds, y_test).item()
        mae = torch.mean(torch.abs(test_preds - y_test)).item()
        r2 = r2_score(y_test, test_preds)
    
    results.append({
        "lr": lr, "hidden": hidden, "epochs": epochs,
        "train_loss": train_loss, "test_loss": test_loss,
        "test_mae": mae, "test_r2": r2, "time_s": round(duration, 2)
    })
    print(f"Done {i+1}/30: MAE={mae:.2f}")

results.sort(key=lambda x: x['test_mae'])

print("\nRank | LR      | Hidden | Epochs | Test MAE    | Test R²   | Time (s)")
print("-" * 70)
for idx, res in enumerate(results[:10]):
    print(f"{idx+1:<4} | {res['lr']:<7} | {res['hidden']:<6} | {res['epochs']:<6} | {res['test_mae']:<11.2f} | {res['test_r2']:<9.4f} | {res['time_s']}")

with open('experiments.json', 'w') as f:
    json.dump(results, f, indent=4)

plt.figure(figsize=(10, 6))
for h in hidden_sizes:
    subset = [r for r in results if r['hidden'] == h]
    subset.sort(key=lambda x: x['lr'])
    lrs = [str(r['lr']) for r in subset]
    maes = [r['test_mae'] for r in subset]
    plt.plot(lrs, maes, marker='o', label=f'Hidden: {h}')

plt.axhline(y=10000, color='r', linestyle='--', label='Target 10k')
plt.xlabel('Learning Rate')
plt.ylabel('Test MAE (JOD)')
plt.title('Hyperparameter Search Summary')
plt.legend()
plt.grid(True)
plt.savefig('experiment_summary.png') 
