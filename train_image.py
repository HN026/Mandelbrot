import torch
import src.models as models
from src.imageDataset import ImageDataset
from torch.utils.data import DataLoader
from torch import optim, nn 
import matplotlib.pyplot as plt 
from src.videomaker import renderModel
from tqdm import tqdm 
import os


image_path = 'DatasetImages/helloworld.png'
hidden_size = 300
num_hidden_layers = 30
batch_size = 8000
lr = 0.001
num_epochs = 30
proj_name = 'helloworld_skipconn'
save_every_n_iterations = 2
scheduler_step = 3

dataset = ImageDataset(image_path)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
resx, resy = dataset.width, dataset.height
linspace = torch.stack(torch.meshgrid(torch.linspace(-1,1,resx), torch.linspace(1,-1,resy)), dim=-1).cuda()
linspace = torch.rot90(linspace, 1, (0,1))
print(linspace.shape)

model = models.SkipConn(hidden_size=hidden_size, num_hidden_layers=num_hidden_layers).cuda()

loss_func = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=0.5)

iteration, frame = 0,0
for epoch in range(num_epochs):
    epoch_loss = 0
    for x, y in tqdm(loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        x,y = x.cuda(), y.cuda()

        y_pred = model(x).squeeze()

        loss = loss_func(y_pred, y)
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
        optimizer.step()

        if iteration % save_every_n_iterations == 0:
            os.makedirs(f'./frames/{proj_name}', exist_ok = True)
            plt.imsave(f'./frames/{proj_name}/{frame:05d}.png', renderModel(model, resx=resx, resy=resy, linspace=linspace), cmap='magma', origin='lower')
            frame += 1
        iteration += 1

    scheduler.step()
    print(f'Epoch {epoch+1}, Average Loss: {epoch_loss/len(loader)}')

os.system(f'ffmpeg -y -r 30 -i ./frames/{proj_name}/%05d.png -c:v libx264 -preset veryslow -crf 0 -pix_fmt yuv420p ./frames/{proj_name}/{proj_name}.mp4')