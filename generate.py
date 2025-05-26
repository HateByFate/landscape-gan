import torch
import os
from models import Generator
from utils import save_images

def generate_images(model_path, num_images=16, output_dir="generated"):
    """
    Генерирует новые изображения с помощью обученной модели
    
    Args:
        model_path (str): Путь к файлу с весами генератора (.pth)
        num_images (int): Количество изображений для генерации
        output_dir (str): Директория для сохранения результатов
    """
    # Создаем директорию для результатов
    os.makedirs(output_dir, exist_ok=True)
    
    # Определяем устройство (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Создаем модель генератора
    generator = Generator().to(device)
    
    # Загружаем веса
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()  # Переключаем в режим оценки
    
    # Генерируем случайный шум
    noise = torch.randn(num_images, 100, 1, 1, device=device)
    
    # Генерируем изображения
    with torch.no_grad():
        fake_images = generator(noise)
    
    # Сохраняем результат
    output_path = os.path.join(output_dir, "generated_landscapes.png")
    save_images(fake_images, output_path, nrow=4)
    print(f"Изображения сохранены в {output_path}")

if __name__ == "__main__":
    # Путь к последней сохраненной модели
    model_path = "output/generator_epoch_200.pth"  # Измените на нужную эпоху
    
    # Генерируем 16 новых изображений
    generate_images(model_path, num_images=16) 
