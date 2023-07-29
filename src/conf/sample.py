import torch
import numpy as np
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, inputs, labels, transforms=None):
        """
        Args:
            inputs (list or numpy array): List or numpy array containing input features.
            labels (list or numpy array): List or numpy array containing corresponding labels.
            transforms (list of callables, optional): List of transformations to be applied on a sample.
        """
        self.inputs = inputs
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get the sample at the specified index
        input_sample = self.inputs[idx]
        label = self.labels[idx]

        # Convert input and label to PyTorch tensors
        input_sample = torch.tensor(input_sample)
        label = torch.tensor(label)

        # Apply transformations if provided
        if self.transforms:
            for transform in self.transforms:
                input_sample, label = transform(input_sample, label)

        return input_sample, label

def normalize_features(inputs, labels):
    inputs = (inputs - torch.mean(inputs, dim=0)) / torch.std(inputs, dim=0)
    return inputs, labels

# Sample data (replace this with your actual data)
inputs = np.random.rand(100, 5)  # Assuming 100 samples with 5 features each
labels = np.random.randint(0, 2, size=(100,))  # Assuming binary labels for each sample

# Create the CustomDataset object with the normalize_features transform
transforms_list = [normalize_features]
dataset = CustomDataset(inputs, labels, transforms=transforms_list)

# Accessing a sample from the dataset
input_sample, label = dataset[0:5]

# Printing the shape of the input tensor and label tensor
print(input_sample.shape, label.shape)
