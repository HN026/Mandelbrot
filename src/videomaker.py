import os, torch
from src.dataset import mandelbrot, smoothMandelbrot, mandelbrotGPU, mandelbrotTensor
from tqdm import tqdm 
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("./captures/images", exist_ok=True)

def renderMandelbrot(resx, resy, xmin=-2.4, xmax=1, yoffset=0, max_depth=50, gpu=False):
    step_size = (xmax-xmin)/resx
    y_start = step_size * resy/2
    ymin = -y_start-yoffset
    ymax = y_start-yoffset
    if not gpu:
        X = np.arange(xmin, xmax, step_size)[:resx]
        Y = np.arange(ymin, ymax, step_size)[:resy]
        im = np.zeros((resy, resx))
        for j, x in enumerate(tqdm(X)):
            for i, y in enumerate(Y):
                im[i, j] = mandelbrot(x,y, max_depth)
        return im
    else:
        return mandelbrotGPU(resx, resy, xmin, xmax, ymin, ymax, max_depth).cpu().numpy()
    
def renderModel(model, resx, resy, xmin=-2.4, xmax=1, yoffset=0, linspace=None, max_gpu=False):
    with torch.no_grad():
        model.eval()
        if linspace is None:
            linspace = generateLinspace(resx, resy, xmin, xmax, yoffset)
        linspace = linspace.cuda()

        if not max_gpu:
            im_slices = []
            for points in linspace:
                im_slices.append(model(points))
            im = torch.stack(im_slices, 0)
        else:
            if linspace.shape != (resx*resy, 2):
                linspace = torch.reshape(linspace, (resx*resy, 2))
            im = model(linspace).squeeze()
            im = torch.reshape(im, (resy, resx))

        im = torch.clamp(im, 0 , 1)
        linspace = linspace.cpu()
        torch.cuda.empty_cache()
        model.train()
        return im.squeeze().cpu().numpy()
    
def generateLinspace(resx, resy, xmin=-2.4, xmax=1, yoffset=0):
    iteration = (xmax-xmin)/resx
    X = torch.arange(xmin, xmax, iteration).cuda()[:resx]
    y_max = iteration * resy/2
    Y = torch.arange(-y_max-yoffset, y_max-yoffset, iteration)[:resy]
    linspace = []
    for y in Y:
        ys = torch.ones(len(X)).cuda() * Y
        points = torch.stack([X, ys], 1)
        linspace.append(points)
    return torch.stack(linspace, 0)