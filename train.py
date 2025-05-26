import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

from models import Generator, Discriminator
from data_loader import get_dataloader
from utils import save_images, plot_losses, weights_init

# Параметры
BATCH_SIZE = 64
IMAGE_SIZE = 64
LATENT_DIM = 100
NUM_EPOCHS = 200
LEARNING_RATE = 0.0002
BETA1 = 0.5
NUM_WORKERS = 4
SAVE_INTERVAL = 10

# Пути
DATA_DIR = "data/landscapes"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    # Инициализация моделей
    netG = Generator(LATENT_DIM).to(device)
    netD = Discriminator().to(device)
    
    # Инициализация весов
    netG.apply(weights_init)
    netD.apply(weights_init)
    
    # Оптимизаторы
    optimizerD = optim.Adam(netD.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
    
    # Функция потерь
    criterion = nn.BCELoss()
    
    # Загрузка данных
    dataloader = get_dataloader(DATA_DIR, BATCH_SIZE, IMAGE_SIZE, NUM_WORKERS)
    
    # Для отслеживания потерь
    g_losses = []
    d_losses = []
    
    # Фиксированный шум для визуализации
    fixed_noise = torch.randn(64, LATENT_DIM, 1, 1, device=device)
    
    print("Начало обучения...")
    
    for epoch in range(NUM_EPOCHS):
        for i, real_images in enumerate(tqdm(dataloader)):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            
            # Метки
            real_label = torch.ones(batch_size, device=device)
            fake_label = torch.zeros(batch_size, device=device)
            
            # Обучение дискриминатора
            netD.zero_grad()
            output = netD(real_images)
            d_loss_real = criterion(output, real_label)
            d_loss_real.backward()
            
            noise = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)
            fake_images = netG(noise)
            output = netD(fake_images.detach())
            d_loss_fake = criterion(output, fake_label)
            d_loss_fake.backward()
            
            d_loss = d_loss_real + d_loss_fake
            optimizerD.step()
            
            # Обучение генератора
            netG.zero_grad()
            output = netD(fake_images)
            g_loss = criterion(output, real_label)
            g_loss.backward()
            optimizerG.step()
            
            # Сохранение потерь
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
            
            if i % 100 == 0:
                print(f'[{epoch}/{NUM_EPOCHS}][{i}/{len(dataloader)}] '
                      f'Loss_D: {d_loss.item():.4f} Loss_G: {g_loss.item():.4f}')
        
        # Сохранение результатов
        if (epoch + 1) % SAVE_INTERVAL == 0:
            with torch.no_grad():
                fake = netG(fixed_noise)
                save_images(fake, f'{OUTPUT_DIR}/fake_samples_epoch_{epoch+1}.png')
                plot_losses(g_losses, d_losses, f'{OUTPUT_DIR}/losses_epoch_{epoch+1}.png')
            
            # Сохранение моделей
            torch.save(netG.state_dict(), f'{OUTPUT_DIR}/generator_epoch_{epoch+1}.pth')
            torch.save(netD.state_dict(), f'{OUTPUT_DIR}/discriminator_epoch_{epoch+1}.pth')

if __name__ == "__main__":
    train() 
