import numpy
import torch
from torch.utils.data import Dataset


def get_torch_dtype(numpy_dtype):
    if numpy_dtype == numpy.float32:
        return torch.float32
    elif numpy_dtype == numpy.float64:
        return torch.float64


def torch_fill_diag(t, number, dtype, device):
    size = t.shape[0]
    mask = torch.eye(size, size, dtype=dtype, device=device).byte()
    return t.masked_fill_(mask, number)


class SimpleDataset(Dataset):
    def __init__(self, samples, labels):
        super(SimpleDataset, self).__init__()
        self.samples = samples
        self.labels = labels

    def __len__(self):
        """
        Denotes the total number of samples
        Returns:

        """
        return len(self.samples)

    def __getitem__(self, index):
        """
            Generates one sample of data.
        Args:
            index: index of sample

        Returns:

        """
        # Select sample
        x = self.samples[index]
        x = torch.tensor(x, dtype=torch.float32)  # TODO: Fixed float32 dtype due to pickling problems in dataloader's workers
        y = self.labels[index]
        y = torch.tensor(y, dtype=torch.float32)  # TODO: Fixed float32 dtype due to pickling problems in dataloader's workers
        return x, y
