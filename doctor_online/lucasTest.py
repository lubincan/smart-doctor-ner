import torch

name = torch.randn(5,3,4)
print(name)
newName = name.transpose(2,1)
print(newName)