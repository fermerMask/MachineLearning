import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

class NetModel(nn.Module):
    def __init__(self, input_size):
        super(NetModel,self).__init__()
        self.size = input_size*input_size
        self.fc1 = nn.Linear(self.size, 1024)
        self.fc2 = nn.Linear(1024,256)
        self.fc3 = nn.Linear(256,10)
    
    def forward(self,x):
        x = x.view(-1,self.size)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def do_train(model,device,loader,criterion,optimizer):
    model.train()
    tot_loss = 0.0
    tot_score = 0.0
    for images, labels in tqdm(loader, desc='train'):
        images, labels = images.to(device),labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        tot_loss += loss.detach().item()
        tot_score += accuracy_score(labels.cpu)

train_dataset = torchvision.datasets.MNIST(root="data", 
                                           train=False, 
                                           transform=torchvision.transforms.ToTensor(), 
                                           download=True)
valid_dataset = torchvision.datasets.MNIST(root="data", 
                                           train=False, 
                                           transform=torchvision.transforms.ToTensor(), 
                                           download=True)

fig, ax = plt.subplots(5,5, figsize=(10,10))

for i in range(25) :
  img, label = train_dataset[i] 
  r, c = i//5, i%5
  ax[r, c].imshow(img.squeeze(), cmap="gray")
  ax[r, c].axis("off")
  ax[r, c].set_title(label)

batch_size = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
