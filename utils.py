import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
import torch.nn as nn

def save_images(images, path, nrow=8):
    """Сохраняет сетку изображений"""
    grid = make_grid(images, nrow=nrow, normalize=True)
    plt.figure(figsize=(15, 15))
    plt.imshow(np.transpose(grid.cpu().numpy(), (1, 2, 0)))
    plt.axis('off')
    plt.savefig(path)
    plt.close()

def plot_losses(g_losses, d_losses, path):
    """Строит график потерь"""
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(path)
    plt.close()

def weights_init(m):
    """Инициализация весов для моделей"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0) 
