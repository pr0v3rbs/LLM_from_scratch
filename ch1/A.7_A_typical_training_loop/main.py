import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = torch.nn.Sequential(
            # 1st hidden layer
            # The input size is num_inputs, and the output size is 30
            torch.nn.Linear(num_inputs, 30),
            # ReLU activation function placed between the hidden layers.
            torch.nn.ReLU(),

            # 2nd hidden layer
            # The number of output nodes of one hidden layer has to match the number of inputs of
            # the next layer.
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            # output layer
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits   # logits are the raw scores output by the last layer

class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    # The __getitem__ method is used to retrieve the data and labels for a given index.
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx] # one_x, one_y
    
    # The __len__ method is used to return the length of the dataset.
    def __len__(self):
        return self.labels.shape[0]

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

X_train = torch.tensor([
    [-1.2, 3.1],
    [-0.9, 2.9],
    [-0.5, 2.6],
    [2.3, -1.1],
    [2.7, -1.5]
])
y_train = torch.tensor([0, 0, 0, 1, 1])

X_test = torch.tensor([
    [-0.8, 2.8],
    [2.6, -1.6],
])
y_test = torch.tensor([0, 1])

train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_test)

train_loader = DataLoader(
    dataset=train_ds,   # The ToyDataset instance created earlier
    batch_size=2,
    shuffle=True,
    num_workers=0,  # Data loading will be done in the main process (Make a bottleneck)
    drop_last=True
)

test_loader = DataLoader(
    dataset=test_ds,
    batch_size=2,
    shuffle=False,  # Not necessary to shuffle
    num_workers=0
)

torch.manual_seed(123)
model = NeuralNetwork(num_inputs=2, num_outputs=2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):
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
              f" | Batch {batch_idx+1:03d}/{len(train_loader):03d}"
              f" | Train Loss: {loss:.2f}")

    model.eval()
    # Insert optimal model evaluation code

### Excercise A.3
### Q: How many parameters does the neural network introduced in listing A.9 have?
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(num_params)  # 752

model.eval()
with torch.no_grad():
    outputs = model(X_train)
print(outputs)  # logits are the raw scores output by the last layer

torch.set_printoptions(sci_mode=False)
probas = torch.softmax(outputs, dim=1)
print(probas)  # probabilities of each class

print(compute_accuracy(model, train_loader))
print(compute_accuracy(model, test_loader))