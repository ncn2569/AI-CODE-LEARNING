import logging

import torch
from torch import nn

logger = logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])

y = torch.tensor([[0.0], [0.0], [0.0], [1.0]])

model = nn.Sequential(
    nn.Linear(2, 4),  # 2 input -> 4 neurons
    nn.ReLU(),
    nn.Linear(4, 1),  # 4 neurons -> 1 output
)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(1000):
    # forward
    prediction = model(X)
    # loss
    loss = loss_fn(prediction, y)
    # back propagate
    optimizer.zero_grad()
    loss.backward()
    # update weights
    optimizer.step()

    if epoch % 100 == 0:
        logger.info(" epoch = %d, loss = %.6f",epoch, loss.item())

logger.info(model(X).detach())
