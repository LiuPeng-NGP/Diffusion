import torch
sigma_max = 80
sigma_data = 0.5
t = torch.arctan(torch.tensor([sigma_max / sigma_data]))
print(t)