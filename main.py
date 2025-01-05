from helpers import batched_tokenizer, convert_digits_to_random_text
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import tqdm

class trainer:
    def __init__(self, learning_rate: float, batch_size: int, num_epochs: int, device : str = 'cuda:0'):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device
        self.model = ...
        self.optimizer = ...
    

    def validate(self):        
        raise NotImplementedError()


    def fit(self):
        transform = transforms.Compose([
            transforms.ToTensor(),               
            transforms.Normalize((0.5,), (0.5,)) 
        ])

        train_dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Training Loop
        # Hint: Do Preprocessing of Labels using helpers
        ...

if __name__ == '__main__':
    trainer(0.001, 256, 5).fit()