# utils.py
# データローダと前処理

import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_image_dataloader(data_dir, batch_size, image_size, grayscale, normalize):
    transform_list = [transforms.Resize(image_size)]
    if grayscale:
        transform_list.append(transforms.Grayscale())
    transform_list.append(transforms.ToTensor())
    if normalize:
        transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]) if grayscale else
                              transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]))
    transform = transforms.Compose(transform_list)

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader
