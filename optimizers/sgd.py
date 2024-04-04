import numpy as np
from scipy import sparse
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from opacus.privacy_engine import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier

class Model(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 1, bias=False)
    
    def forward(self, x):
        return self.fc(x)

def fit(X, y, type=None, batch_size=None, lr=None, T=None, epsilon=None, delta=None):
    """
    Fit function for stochastic gradient descent using opacus. 
    Employed as a baseline in the study. 
    """
    assert isinstance(epsilon, (float, int))
    if delta is None: delta = (1 / X.shape[0]) ** 2
    n, p = X.shape
    model = Model(p)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    X_tensor = torch.tensor(X).float()
    y_tensor = torch.tensor(y).float()
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size)
    noise_multiplier = get_noise_multiplier(target_epsilon=epsilon, target_delta=delta, sample_rate=batch_size / n, steps=T, epsilon_tolerance=1e-3)
    privacy_engine = PrivacyEngine()
    model, optimizer, dataloader = privacy_engine.make_private(module=model, optimizer=optimizer, data_loader=dataloader, noise_multiplier=noise_multiplier, max_grad_norm=1.0)
    if type == 'linear':
        criterion = nn.MSELoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    i = 0
    while i < T:
        for (X_batch, y_batch) in dataloader:
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out.flatten(), y_batch)
            loss.backward()

            i += 1
            if i >= T:
                break

    return model.fc.weight.flatten().detach().cpu().numpy()