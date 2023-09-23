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

class VideoMaker:
    def __init__(self, name='autosave', fps=30, dims=(100,100), capture_rate=10, shots=None,
                 max_gpu=False, cmap='magma'):
        self.name = name
        self.dims = dims
        self.capture_rate = capture_rate
        self.max_gpu = max_gpu
        self._xmin = -2.4
        self._xmax = 1
        self._yoffset = 0
        self.shots = 0
        self._yoffset = 0
        self.shots = shots
        self.cmap = cmap
        self.fps = fps
        os.makedirs(f'./frames/{self.name}', exist_ok = True)

        self.linspace = generateLinspace(self.dims[0], self.dims[1], self._xmin, self._xmax, self._yoffset)
        if max_gpu:
            self.linspace = torch.reshape(self.linspace, (dims[0]*dims[1], 2))

        self.frame_count = 0

    def generateFrame (self, model):
        if self.shots is not None and len(self.shots) > 0 and self.frame_count >= self.shots[0]['frame']:
            shot = self.shots.pop(0)
            self._xmin = shot["xmin"]
            self._xmax = shot["xmax"]
            self._yoffset = shot["yoffset"]
            if len(shot) > 4:
                self.capture_rate = shot["capture_rate"]
            self.linspace = generateLinspace(self.dims[0], self.dims[1], self._xmin, self._xmax, self._yoffset)

        im = renderModel(model, self.dims[0], self.dims[1], linspace=self.linspace, max_gpu = self.max_xgpu)
        plt.imsave(f'./frames/{self.name}/{self.frame_count:05d}.png', im, cmap = self.cmap)

    def generateVideo(self):
        os.system(f'ffmpeg -y -r {self.fps} -i ./frames/{self.name}%05d.png -c:v libx264 -preset veryslow -crf 0 -pix_fmt yuv420p ./frames/{self.name}/self.name.mp4')