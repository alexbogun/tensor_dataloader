from tensor_dataloader import *
import sys, os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets
import torch.optim as optim
import time as tm

# Torch init
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MNIST Dataset 
dataset = datasets.MNIST(root='./data/mnist/', train=False, download=True)  

# get Tensors
images = torch.flatten(dataset.data[:].float(),start_dim=1).to(device)
label = dataset.targets[:].to(device)

# TensorDataset
dataset = torch.utils.data.TensorDataset(images, label)
dataloader = FastTensorDataLoader(dataset, batch_size=1024, shuffle=True)

input_size = 784
hidden_sizes = [128, 64]
output_size = 10

model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1)).to(device)

optimizer = optim.Adam(model.parameters())

n_epochs = 1000
tt= tm.time()

for epoch in range(n_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

    print('Train Epoch: {}/{} \tLoss: {:.6f}'.format(epoch+1, n_epochs, loss.item()))

print("Training using FastTensorDataLoader took:", np.round(tm.time()-tt,2),"seconds")
