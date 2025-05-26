import torch
import numpy as np
from torch.utils.data import DataLoader
from pytorch_fid import fid_score
import os
from models import Generator
from data_loader import get_dataloader
from utils import save_images
from torchvision.utils import save_image
import torchvision.transforms.functional as F
from PIL import Image
import torchvision.transforms as transforms

def preprocess_images(input_dir, output_dir, size=64):
    """Предобработка изображений для FID"""
    os.makedirs(output_dir, exist_ok=True)
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    for img_name in os.listdir(input_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_dir, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img)
                # Денормализуем и сохраняем
                img_tensor = (img_tensor + 1) / 2
                F.to_pil_image(img_tensor).save(os.path.join(output_dir, f'processed_{img_name}'))
            except Exception as e:
                print(f"Ошибка при обработке {img_name}: {e}")

def calculate_fid(real_dir, generated_dir, batch_size=64, device='cuda'):
    """
    Вычисляет FID между реальными и сгенерированными изображениями
    
    Args:
        real_dir (str): Путь к папке с реальными изображениями
        generated_dir (str): Путь к папке с сгенерированными изображениями
        batch_size (int): Размер батча для вычисления
        device (str): Устройство для вычислений
    """
    # Создаем директории
    processed_real_dir = os.path.join(generated_dir, 'processed_real')
    os.makedirs(generated_dir, exist_ok=True)
    
    # Предобработка реальных изображений
    print("Предобработка реальных изображений...")
    preprocess_images(real_dir, processed_real_dir)
    
    # Загружаем модель генератора
    generator = Generator().to(device)
    try:
        # Пробуем загрузить последнюю сохраненную модель
        model_files = [f for f in os.listdir('output') if f.startswith('generator_epoch_')]
        if not model_files:
            raise FileNotFoundError("Не найдены сохраненные модели")
        latest_model = max(model_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        generator.load_state_dict(torch.load(os.path.join('output', latest_model), map_location=device))
        print(f"Загружена модель: {latest_model}")
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        return None
    
    generator.eval()
    
    # Генерируем изображения
    num_images = len(os.listdir(processed_real_dir))
    print(f"Генерация {num_images} изображений...")
    with torch.no_grad():
        for i in range(0, num_images, batch_size):
            batch_size_actual = min(batch_size, num_images - i)
            # Изменяем размерность шума для соответствия архитектуре генератора
            noise = torch.randn(batch_size_actual, 100, 1, 1, device=device)
            fake_images = generator(noise)
            
            # Сохраняем сгенерированные изображения
            for j, img in enumerate(fake_images):
                # Нормализуем изображение из [-1, 1] в [0, 1]
                img = (img + 1) / 2
                # Сохраняем изображение
                F.to_pil_image(img.cpu()).save(os.path.join(generated_dir, f'fake_{i+j}.png'))
    
    try:
        # Вычисляем FID
        print("Вычисление FID...")
        fid_value = fid_score.calculate_fid_given_paths(
            [processed_real_dir, generated_dir],
            batch_size=batch_size,
            device=device,
            dims=2048
        )
        return fid_value
    except Exception as e:
        print(f"Ошибка при вычислении FID: {e}")
        return None

if __name__ == "__main__":
    # Пути к директориям
    real_dir = "data/landscapes"
    generated_dir = "generated_for_fid"
    
    # Вычисляем FID
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fid = calculate_fid(real_dir, generated_dir, device=device)
    
    if fid is not None:
        print(f"FID score: {fid:.2f}")
    else:
        print("Не удалось вычислить FID score")
