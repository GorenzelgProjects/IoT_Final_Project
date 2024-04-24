import numpy as np
import torch
from torch import nn
import torch.multiprocessing
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import torchvision
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from timeit import default_timer as timer

# Define a neural network class that inherits from PyTorch nn.Module.
class neuralNetworkV1(nn.Module):
    # The __init__ method is used to declare the layers that will be used in the forward pass.
    def __init__(self):
        super().__init__() # required because our class inherit from nn.Module
        # First convolutional layer with 3 input channels for RGB images, 16 outputs (filters).
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        # Second convolutional layer with 16 input channels to capture features from the previous layer, 16 outputs (filters).
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        # Third and fourth convolutional layers with 16 and 10 output channels respectively.
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(10, 10, kernel_size=3, stride=2, padding=1)
        # Max pooling layer to reduce feature complexity.
        self.pooling = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # ReLU activation function for introducing non-linearity.
        self.relu = nn.ReLU()
        # Flatten the 2D output from the convolutional layers for the fully connected layer.
        self.flatten = nn.Flatten()
        # Fully connected layer connecting to 1D neurons, with 3 output features for 3 classes.
        self.linear = nn.Linear(in_features=480, out_features=3)
    
    # define how each data sample will propagate in each layer of the network
    def forward(self, x: torch.Tensor):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pooling(x)
        x = self.relu(self.conv3(x))
        x = self.pooling(x)
        x = self.relu(self.conv4(x))
        x = self.flatten(x)
        try:
            x = self.linear(x)
        except Exception as e:
            print(f"Error : Linear block should take support shape of {x.shape} for in_features.")
        return x