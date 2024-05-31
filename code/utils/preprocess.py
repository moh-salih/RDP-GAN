import torch
from utils.dataset import AdultDataset, CustomMNIST


def load_dataset(dataset_name, batch_size, shuffle):
    # Decide which dataset to train on
    if dataset_name == 'mnist':
        dataset = CustomMNIST()
    elif dataset_name == 'adult':
        dataset = AdultDataset()

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader, dataset.num_features