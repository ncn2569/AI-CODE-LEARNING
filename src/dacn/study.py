from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 2)

    def forward(self, x):
        return self.fc(x)


# ===== Dữ liệu của 2 clients =====
X1 = torch.randn(100, 2)
y1 = (X1[:, 0] + X1[:, 1] > 0).long()

X2 = torch.randn(100, 2) + 1
y2 = (X2[:, 0] + X2[:, 1] > 2).long()

clients = [(X1, y1), (X2, y2)]


def train_client(global_model, X, y):
    model = deepcopy(global_model)

    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(5):
        optimizer.zero_grad()

        output = model(X)
        loss = loss_fn(output, y)

        loss.backward()
        optimizer.step()

    return model.state_dict()


def fedavg(client_weights):
    avg_weights = deepcopy(client_weights[0])

    for key in avg_weights:
        avg_weights[key] = sum(w[key] for w in client_weights) / len(client_weights)

    return avg_weights


global_model = Model()
for round_id in range(10):
    client_updates = []

    for X, y in clients:
        weights = train_client(global_model, X, y)
        client_updates.append(weights)

    new_global_weights = fedavg(client_updates)

    global_model.load_state_dict(new_global_weights)

    print(f"Round {round_id + 1} completed")
