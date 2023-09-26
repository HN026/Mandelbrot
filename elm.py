import torch 
import torch.nn as nn
from src.videomaker import renderModel
from src.dataset import MandelbrotDataSet
from src.imageDataset import ImageDataset
import matplotlib.pyplot as plt
import numpy as np

class ELM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ELM, self).__init__()
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim, bias=False)
        self.activation = nn.SELU()

        nn.init.uniform_(self.hidden_layer.weight)
        self.hidden_layer.weight.requires_grad = False

    def forward(self, x):
        x = self.activation(self.hidden_layer(x))
        x = self.output_layer(x)
        return x
    
dataset = ImageDataset('./DatasetImages/blob_small.png')
x = torch.stack([dataset[i][0] for i in range(len(dataset))])
y = torch.unsqueeze(torch.stack([dataset[i][1] for i in range(len(dataset))]), 1)
print(x.shape, y.shape)

model = ELM(2,1000, 1)

def evaluate_model(model, x, y):
    with torch.no_grad():
        outputs = model(x)
        mse = ((outputs - y) ** 2).mean().item()
    return mse

print('Before training, MSE: {:f}'.format(evaluate_model(model, x, y)))

H = model.hidden_layer(x)
H = model.activation(H)

print(H.shape)
H_pinv = torch.linalg.pinv(H)
print(H_pinv.shape, y.shape)
output_weights = torch.mm(H_pinv, y)

model.output_layer.weight.data = output_weights.view(model.output_layer.weight.data.size())

print('After Training, MSE: {:f}'.format(evaluate_model(model, x, y)))

model.cuda()

resx, resy = dataset.width, dataset.height
linspace = torch.stack(torch.meshgrid(torch.linspace(-1,1,resx), torch.linspace(1,-1,resy)), dim=-1).cuda()
linspace = torch.rot90(linspace, 1, (0,1))
plt.imshow(renderModel(model, resx=resx, resy=resy, linspace=linspace), cmap='magma', origin='lower')
plt.show()