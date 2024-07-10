import torchvision
import subprocess
import torch
import os
from torch.utils.data import random_split

def get_butterflies256_dataset(train: bool, root: str = "data/butterflies256", transform=None):

    
    if not os.path.exists(os.path.join(root)):
        print("Downloading and extracting butterflies256 dataset...")
        original_dir = os.getcwd()
        os.chdir(os.path.dirname(root))
        try:
            subprocess.run(['gdown', '1FnzQLDPs-IlTTEr14YyENKjTYqZfn8mS'])
            subprocess.run(['tar', '-xf', 'butterflies256.tar.gz'])
            os.remove('butterflies256.tar.gz')
            print("Download and extraction complete.")
        finally:
            os.chdir(original_dir)
    else:
        pass

    dataset = torchvision.datasets.ImageFolder(root, transform=transform)
    
    # Set the random seed for reproducibility
    torch.manual_seed(42)

    # Splitting the dataset into train and test sets
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    if train:
        return train_dataset
    else:
        return test_dataset


def get_dataset(name, train):
    if name == 'fmnist':
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.Resize(32),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5)),
        ])
        return torchvision.datasets.FashionMNIST(root=f'ImageNet64/dataset/{name}', train=train, download=True, transform=transform)
    elif name == 'mnist':
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(32),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5)),
        ])
        return torchvision.datasets.MNIST(root=f'ImageNet64/dataset/{name}', train=train, download=True, transform=transform)
    elif name == 'cifar10':
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5)),
        ])
        return torchvision.datasets.CIFAR10(root=f'ImageNet64/dataset/{name}', train=train, download=True, transform=transform)
    elif name == 'butterflies256':
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((32,32)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5)),
        ])
        return get_butterflies256_dataset(root=f'ImageNet64/dataset/{name}', train=train, transform=transform)


    
    raise ValueError(f'Dataset {name} not found')
