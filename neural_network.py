import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

"""device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")"""


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(63, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        pred_probab = nn.Softmax(dim=1)(logits)
        return pred_probab

def train(model, x_train_loader, y_train_loader, optimizer, device, loss_fn = nn.MSELoss()):
    NUMBER_OF_EPOCHS = 80
    for epoch in range(NUMBER_OF_EPOCHS):
        accumulated_loss = 0
        accumulated_accuracy = 0
        for (x_train, y_train) in zip(x_train_loader, y_train_loader):
            
            x_train = x_train.to(device).to(torch.float32)
            y_train = y_train.to(device).to(torch.float32)

            model.zero_grad()
            pred_probab = model(x_train)
            
            loss = loss_fn(pred_probab, y_train)
            
            accumulated_loss += loss.item()
            accumulated_accuracy += (1-torch.abs(pred_probab - y_train).mean().item())
            loss.backward() 
            optimizer.step()
            
            #print(f"Average loss epoch {epoch}:", accumulated_loss/x_train.shape[1])
            #print(f"Average accuracy epoch {epoch}:", accumulated_accuracy/x_train.shape[1])
            
def test(model, x_test_loader, y_test_loader, device):
    classification_results = []
    with torch.no_grad():
        i = 0
        for (x_test, y_test) in tqdm(zip(x_test_loader, y_test_loader),desc="Testing"):
            
            x_test = x_test.to(device).to(torch.float32)
            y_test = y_test.to(device).to(torch.float32)
            
            pred_probab = model(x_test)
            
            #loss = loss_fn.item(pred_probab, y_test)
            #test_loss = loss.item()
            
            y_pred = pred_probab.argmax(1)
            #y_test = y_test.argmax(1)
            
            classification_results.append([y_test.item(), y_pred.item()])
            
    return classification_results
            

def one_hot_encoder(y):
    encoded_y = torch.zeros(10)
    encoded_y[int(y)-1] = 1
    return encoded_y