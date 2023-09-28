from src.videomaker import renderMandelbrot, renderModel, VideoMaker
from src.training import train
from src.dataset import MandelbrotDataSet
from src import models
import matplotlib.pyplot as plt
import torch

def example_render():
    image = renderMandelbrot(3840, 2160, max_depth=500, gpu=True)
    plt.imsave('./captures/images/mandel_gpu.png', image, vmin=0, vmax=1, cmap='gist_heat')

def example_train():
    print("Initializing Model...")
    model = models.SkipConn(300,50).cuda()

    dataset = MandelbrotDataSet(2000000, gpu=True)
    eval_dataset = MandelbrotDataSet(100000, gpu=True)

    train(model, dataset, 10, batch_size=10000, eval_dataset=eval_dataset, oversample=0.1, use_scheduler=True, snapshots_every=50)

def example_render_model():
    linmap = models.CenteredLinearMap(x_size=torch.pi*2, y_size=torch.pi*2)
    model = models.Fourier(256, 400, 50, linmap=linmap)
    model.load_state_dict(torch.load('./models/Jun04_00-34-51_xerxes-u.pt'))
    model.cuda()
    image = renderModel(model, 7680, 4320, max_gpu=False)
    plt.imsave('./captures/images/Jun04_00-34-51_xerxes-u.png', image, vmin=0, vmax=1, cmap='inferno')
    plt.show()

def example_train_capture():
    shots = [
        {'frame':5, "xmin":-2.5, "xmax":1, "yoffset":0, "capture_rate":8},
        {'frame':10, "xmin":-1.8, "xmax":-0.9, "yoffset":0.2, "capture_rate":16},
    ]

    vidmaker = VideoMaker('test', dims=(960,544), capture_rate=5, shots=shots, max_gpu=True)
    linmap = models.CenteredLinearMap(x_size=torch.pi*2, y_size=torch.pi*2)
    model = models.Fourier(256,400,50, linmap=linmap)
    dataset = MandelbrotDataSet(2000000, max_depth=1500, gpu=True)
    train(model, dataset, 1, batch_size=8000, use_scheduler=True, oversample=0.1, snapshots_every=500, vm=vidmaker)

def create_dataset():
    dataset = MandelbrotDataSet(100000, max_depth=50, gpu=True)
    dataset.save('1M_50_test')

if __name__ == "__main__":
    example_train_capture()