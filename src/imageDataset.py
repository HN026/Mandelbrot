from PIL import Image  #Pytorch Image Library
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class ImageDataset(Dataset):
    def __init__(self, image_path):
        """Load Image, convert to grayscale and scale pixel values to [0,1]"""
        self.image = Image.open(image_path).convert('L')
        self.image = ToTensor()(self.image)
        
        """Get Image Dimensions"""
        self.height, self.width = self.image.shape[1:]

    def __len__(self):
        return self.height * self.width     # Length of the tensor
    
    def __getitem__(self, idx):
        """Convert flat index to 2D coordinates"""
        row = idx // self.width
        col = idx % self.width

        """Scale coordinates to [-1, 1]"""
        input = torch.tensor([col/(self.width/2)-1, (self.height-row)/(self.height/2)-1])

        """Gete pixel value"""
        output = self.image[0, row, col]

        return input, output
    
    def display_image(self):
        """Uses the getitem method to get each pixel value and displays the final image, used for debugging purposes"""
        image = torch.zeros((self.height, self.width))
        for i in range(len(self)):
            row = i // self.width
            col = i % self.width
            image[row, col] = self[i][1]
        plt.imshow(image, cmap='gray')
        plt.show()