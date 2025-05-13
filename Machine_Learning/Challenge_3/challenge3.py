import torch
import matplotlib.pyplot as plt
import pandas as pd
import torchvision

print(torch.cuda.is_available())

device = torch.device("cuda")
x = torch.rand(5,5).to(device)
print(x)