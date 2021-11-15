from PIL import Image
import os
import torch
import torchvision
import numpy as np
from torch import nn



class MnistDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        # Here we will store all of our samples in format (image_path, label)
        self.datapoints = []

        dirs = os.listdir(path)
        for file in dirs:
            current_dir = os.path.join(path, file)
            for ndfile in os.listdir(current_dir):
                mytuple = (os.path.join(current_dir, ndfile), file)
                self.datapoints.append(mytuple)

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, index):
        image_path = self.datapoints[index][0]
        img = np.array(Image.open(image_path)).astype(np.float32)
        img = img.reshape(1, 28, 28)
        img = (img - np.min(img)) / (np.max(img) - np.min(img))

        return img, int(self.datapoints[index][1])

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # Layers
        self.conv_layer1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv_layer2 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv_layer3 = nn.Conv2d(16, 32, 3, padding=1)
        self.pooling = nn.MaxPool2d(2)

        self.final_layer1 = nn.Linear(288, 128)
        self.final_layer2 = nn.Linear(128, 10)

        # Activations
        self.activation_hidden = nn.LeakyReLU()
        self.activation_final = nn.LogSoftmax(dim=1)

        # batch norm layers
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)

    def forward(self, x):
        # Feature extraction
        x = self.conv_layer1(x)
        x = self.bn1(x)
        x = self.activation_hidden(x)
        x = self.pooling(x)

        x = self.conv_layer2(x)
        x = self.bn2(x)
        x = self.activation_hidden(x)
        x = self.pooling(x)

        x = self.conv_layer3(x)
        x = self.bn3(x)
        x = self.activation_hidden(x)
        x = self.pooling(x)

        # Classification
        x = x.view(x.shape[0], -1)

        x = self.final_layer1(x)
        x = self.activation_hidden(x)

        x = self.final_layer2(x)
        x = self.activation_final(x)

        return x