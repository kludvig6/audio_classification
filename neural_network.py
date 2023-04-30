import torch
from torch import nn
from tqdm import tqdm

class NeuralNetwork(nn.Module):
    def __init__(self, _num_features):
        super().__init__()
        self.num_features = _num_features
        self.convolution_stack = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

        )
        self.linear_stack = nn.Sequential(        
            nn.Linear(896, 256),
            nn.Dropout(0.2),
            nn.ReLU(),
            
            nn.Linear(256, 128),
            nn.Dropout(0.2),
            nn.ReLU(),
            
            nn.Linear(128, 64),
            nn.Dropout(0.2),
            nn.ReLU(),
            
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # add a channel dimension
        x = self.convolution_stack(x)
        x = x.view(x.size(0), -1)
        logits = self.linear_stack(x)
        pred_probab = nn.Softmax(dim=1)(logits)
        return pred_probab

def train(model, x_train_loader, y_train_loader, optimizer, device, loss_fn = nn.CrossEntropyLoss()):
    NUMBER_OF_EPOCHS = 792
    accumulated_loss = 0
    accumulated_accuracy = 0
    for _ in range(NUMBER_OF_EPOCHS):
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
            
    print(f"Average loss:", 100*accumulated_loss/(NUMBER_OF_EPOCHS*x_train.shape[1]))
    print(f"Average accuracy:", 100*accumulated_accuracy/(NUMBER_OF_EPOCHS*x_train.shape[1]))
            
def test(model, x_test_loader, y_test_loader, device):
    classification_results = []
    with torch.no_grad():
        for (x_test, y_test) in tqdm(zip(x_test_loader, y_test_loader),desc="Testing"):
            x_test = x_test.to(device).to(torch.float32)
            y_test = y_test.to(device).to(torch.float32)
            
            pred_probab = model(x_test)
            y_pred = pred_probab.argmax(1)
            classification_results.append([y_test.item(), y_pred.item()])     
    return classification_results  

def one_hot_encoder(y):
    encoded_y = torch.zeros(10)
    encoded_y[int(y)] = 1
    return encoded_y