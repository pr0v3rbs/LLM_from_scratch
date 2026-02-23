import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import prev_network
import prev_loader

# A.10 A function to compute the prediction accuracy of a model
def compute_accuracy(model, dataloader):
    model = model.eval()
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):
        with torch.no_grad():
            logits = model(features)

        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)

    return (correct / total_examples).item()

### A.9 GPU
print(torch.cuda.is_available())

tensor_1 = torch.tensor([1., 2., 3.])
tensor_2 = torch.tensor([4., 5., 6.])
print(tensor_1 + tensor_2)

tensor_1 = tensor_1.to("cuda")
tensor_2 = tensor_2.to("cuda:0")
print(tensor_1 + tensor_2)

# this will make an error cause different device type
#tensor_1 = tensor_1.to("cpu")
#tensor_2 = tensor_2.to("cuda")
#print(tensor_1 + tensor_2)

torch.manual_seed(123)
model = prev_network.NeuralNetwork(num_inputs=2, num_outputs=2)

# Method 1: using GPU
device = torch.device("cuda")   # Defines a device variable that defaults to a GPU

# Method 2.1: using Normal CPU (Normal PC, MPS(Metal Performance Shaders))
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Method 2.2: using macOS CPU MPS(Metal Performance Shaders)
#device = torch.device( "mps" if torch.backends.mps.is_available() else "cpu" )

model = model.to(device)        # Transfers the model onto the GPU

optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    for batch_idx, (features, labels) in enumerate(prev_loader.train_loader):
        features, labels = features.to(device), labels.to(device)   # Transfers the data onto the GPU
        # Forward pass
        logits = model(features)
        # Compute the loss
        loss = F.cross_entropy(logits, labels)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        # Update the weights
        optimizer.step()

        ### LOGGING
        print(f"Epoch {epoch+1:03d}/{num_epochs}"
              f" | Batch {batch_idx+1:03d}/{len(prev_loader.train_loader):03d}"
              f" | Train Loss: {loss:.2f}")

    model.eval()
    # Insert optimal model evaluation code