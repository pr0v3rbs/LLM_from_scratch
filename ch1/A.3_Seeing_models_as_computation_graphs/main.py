import torch
import torch.nn.functional as F

y = torch.tensor([1.0])     # True label
x1 = torch.tensor([1.1])    # Input feature
w1 = torch.tensor([2.2])    # Weight parameter
b = torch.tensor([0.0])     # Bias unit

z = x1 * w1 + b         # Net input
a = torch.sigmoid(z)    # Activation and output

loss = F.binary_cross_entropy(a, y)

print("x1:", x1)
print("w1:", w1)
print("a:", a)
print("loss:", loss)