import torch

a = torch.randint(0,255,(4,32,32))
b = torch.randint(0,255,(4,32,32))
inputs = torch.randint(0,255,(128,4,32,32))
c = a == b
print(c.shape)
input_vec = inputs[:, (a == b)].T 
print(input_vec.shape)