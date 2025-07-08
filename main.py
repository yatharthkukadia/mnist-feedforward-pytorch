import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
'''
1-download dataset
2-create a dataloader
3-build a model
4-train
5-save trained model
'''
#Built Models
class FeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            nn.Linear(28*28,256),
            nn.ReLU(),
            nn.Linear(256,10)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self,input_data):
        flatten_data = self.flatten(input_data)
        logits = self.dense_layers(flatten_data)
        predictions = self.softmax(logits)
        return predictions

def download_mnist_datasets():
    train_data = datasets.MNIST(
        root='data',
        download=True,
        train=True,
        transform=ToTensor()
    )
    validation_data = datasets.MNIST(
        root='data',
        download=True,
        train=False,
        transform=ToTensor()
    )
    return train_data,validation_data

train_data, _ = download_mnist_datasets()

def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        #calculate loss
        predictions = model(inputs)
        loss = loss_fn(predictions,targets)

        #backpropogate loss and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Loss:{loss.item()}')
def train(model, data_loader, loss_fn, optimizer, device,epoch):
    for i in range(epoch):
        print(f'Epoch {i+1}')
        train_one_epoch(model, data_loader, loss_fn, optimizer, device)
        print('-----------------------------')
    print('Training is Done.')


BATCH_SIZE=128
Epoch = 10
Learning_Rate = 0.001
train_data_loader = DataLoader(train_data,batch_size=BATCH_SIZE)

if torch.cuda.is_available():
    device='cuda'
else:
    device='cpu'

feed_forward_net = FeedForwardNet().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(feed_forward_net.parameters(),
                            lr=Learning_Rate)

train(feed_forward_net, train_data_loader,loss_fn, optimizer, device,epoch=Epoch)

torch.save(feed_forward_net.state_dict(),'feedforwardnet.pth')
print('Model trained and stored at feedforwardnet.pth')

print(f'Using {device} device')