import torch

print(torch.cuda.is_available())
print(torch.backends.mps.is_available())
print(torch.backends.mps.is_built())