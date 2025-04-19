import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

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

print(X_train, y_train)
print(X_test, y_test)

train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_test)

print(len(train_ds))

torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_ds,   # The ToyDataset instance created earlier
    batch_size=2,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    dataset=test_ds,
    batch_size=2,
    shuffle=False,  # Not necessary to shuffle
    num_workers=0
)

print("train_loader:", len(train_loader))
for idx, (x, y) in enumerate(train_loader):
    print(f"Batch {idx+1}:", x, y)

print("test_loader:", len(test_loader))
for idx, (x, y) in enumerate(test_loader):
    print(f"Batch {idx+1}:", x, y)

train_loader = DataLoader(
    dataset=train_ds,   # The ToyDataset instance created earlier
    batch_size=2,
    shuffle=True,
    num_workers=0,  # Data loading will be done in the main process (Make a bottleneck)
    drop_last=True
)

print("train_loader:", len(train_loader))
for idx, (x, y) in enumerate(train_loader):
    print(f"Batch {idx+1}:", x, y)