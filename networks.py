import torch.nn as nn
import torch
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes):
        """
        Initializes a CNN (Convolutional Neural Network) model.

        Args:
            num_classes (int): The number of classes in the classification task.
        """
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # Additional Conv2d layer
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(32 * 28 * 28, num_classes)

    def forward(self, x):
        """
        Performs forward pass of the input through the CNN model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

# Taken from previous assignment in case of use
class FullyConnectedNetwork(nn.Module): 
    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Initializes a fully connected network.

        Args:
            input_dim (int): The dimension of the input data.
            hidden_dim (int): The dimension of the hidden layer.

        Raises:
            AssertionError: If input_dim is not a positive integer or hidden_dim is not an integer greater than 1.
        """
        super(FullyConnectedNetwork, self).__init__()
        assert input_dim > 0, "Input dimension must be a positive integer"
        assert hidden_dim > 1, "Hidden dimensions must be an integer greater than 1"
        self.linear1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.linear2 = nn.Linear(in_features=hidden_dim, out_features=round(hidden_dim // 2))
        self.linear3 = nn.Linear(in_features=round(hidden_dim // 2), out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Performs forward pass through the network.

        Args:
            x: The input data.

        Returns:
            The output of the network.
        """
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x