import torch
from torch.utils.data import Dataset

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