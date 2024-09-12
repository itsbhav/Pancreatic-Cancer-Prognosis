import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.fc(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.fc(x)

z_dim = 100
data_dim = 30  


G = Generator(z_dim, data_dim)
D = Discriminator(data_dim)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=0.0002)
optimizer_D = optim.Adam(D.parameters(), lr=0.0002)


epochs = 10000
batch_size = 64
for epoch in range(epochs):
    
    z = torch.randn(batch_size, z_dim)
    fake_data = G(z)
    
    
    real_data = torch.randn(batch_size, data_dim) 
    D_real = D(real_data)
    D_fake = D(fake_data)
    
    loss_D_real = criterion(D_real, torch.ones(batch_size, 1))
    loss_D_fake = criterion(D_fake, torch.zeros(batch_size, 1))
    loss_D = loss_D_real + loss_D_fake
    
    optimizer_D.zero_grad()
    loss_D.backward()
    optimizer_D.step()
    
    z = torch.randn(batch_size, z_dim)
    fake_data = G(z)
    D_fake = D(fake_data)
    
    loss_G = criterion(D_fake, torch.ones(batch_size, 1))
    
    optimizer_G.zero_grad()
    loss_G.backward()
    optimizer_G.step()
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss D: {loss_D.item()}, Loss G: {loss_G.item()}")
