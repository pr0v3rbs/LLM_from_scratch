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

torch.manual_seed(123)
model = prev_network.NeuralNetwork(num_inputs=2, num_outputs=2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (features, labels) in enumerate(prev_loader.train_loader):
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

### Excercise A.3
### Q: How many parameters does the neural network introduced in listing A.9 have?
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(num_params)  # 752

model.eval()
with torch.no_grad():
    outputs = model(prev_loader.X_train)
print(outputs)  # logits are the raw scores output by the last layer

torch.set_printoptions(sci_mode=False)
probas = torch.softmax(outputs, dim=1)
print(probas)  # probabilities of each class

print(compute_accuracy(model, prev_loader.train_loader))
print(compute_accuracy(model, prev_loader.test_loader))