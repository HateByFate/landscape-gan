import torch
import numpy as np
from torch.utils.data import DataLoader
from pytorch_fid import fid_score
import os
from models import Generator
from data_loader import get_dataloader
from utils import save_images

def calculate_fid(real_dir, generated_dir, batch_size=64, device='cuda'):
    """
    Вычисляет FID между реальными и сгенерированными изображениями
    
    Args:
        real_dir (str): Путь к папке с реальными изображениями
        generated_dir (str): Путь к папке с сгенерированными изображениями
        batch_size (int): Размер батча для вычисления
        device (str): Устройство для вычислений
    """
    # Создаем директорию для сгенерированных изображений, если её нет
    os.makedirs(generated_dir, exist_ok=True)
    
    # Загружаем модель генератора
    generator = Generator().to(device)
    generator.load_state_dict(torch.load('output/generator_epoch_200.pth', map_location=device))
    generator.eval()
    
    # Генерируем изображения
    num_images = len(os.listdir(real_dir))
    with torch.no_grad():
        for i in range(0, num_images, batch_size):
            batch_size_actual = min(batch_size, num_images - i)
            noise = torch.randn(batch_size_actual, 100, 1, 1, device=device)
            fake_images = generator(noise)
            
            # Сохраняем сгенерированные изображения
            for j, img in enumerate(fake_images):
                save_images(img.unsqueeze(0), 
                          os.path.join(generated_dir, f'fake_{i+j}.png'))
    
    # Вычисляем FID
    fid_value = fid_score.calculate_fid_given_paths(
        [real_dir, generated_dir],
        batch_size=batch_size,
        device=device,
        dims=2048
    )
    
    return fid_value

if __name__ == "__main__":
    # Пути к директориям
    real_dir = "data/landscapes"
    generated_dir = "generated_for_fid"
    
    # Вычисляем FID
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fid = calculate_fid(real_dir, generated_dir, device=device)
    
    print(f"FID score: {fid:.2f}")
    
    # Интерпретация результата:
    # FID < 50: отличное качество
    # 50 < FID < 100: хорошее качество
    # 100 < FID < 200: среднее качество
    # FID > 200: низкое качество 
