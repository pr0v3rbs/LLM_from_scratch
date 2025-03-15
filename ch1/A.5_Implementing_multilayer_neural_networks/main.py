import torch

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

model = NeuralNetwork(50, 3)
print(model)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of trainerable parameters: ", num_params)

print(model.layers[0].weight)

print(model.layers[0].weight.shape)