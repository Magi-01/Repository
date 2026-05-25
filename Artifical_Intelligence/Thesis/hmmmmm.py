import torch

checkpoint = torch.load("best_model.pt", map_location="cpu")

print("Keys in checkpoint:", checkpoint.keys())
print(checkpoint['fc.0.weight'][:5])
print(checkpoint['fc.0.bias'][:5])
print("\n")
print(checkpoint['fc.2.weight'][:5])
print(checkpoint['fc.2.bias'][:5])
print("\n")
print(checkpoint['fc.4.weight'][:5])
print(checkpoint['fc.4.bias'][:5])