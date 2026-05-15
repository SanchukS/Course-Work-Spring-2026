import os
import zipfile
import kaggle
from pathlib import Path


def download_and_extract_data(path: Path):
    """
    Проверяет наличие нужных файлов, если их нет - скачивает датасет с Kaggle и распаковывает.
    """
    # 1. Определяем переменные
    dataset_slug = 'frtgnn/dunnhumby-the-complete-journey'
    zip_file_name = 'dunnhumby-the-complete-journey.zip'
    zip_file_path = path / zip_file_name
    files_to_check = ['transaction_data.csv', 'product.csv']
    paths_to_check = [path / file_name for file_name in files_to_check]

    # 2. Проверяем, есть ли уже нужные файлы
    all_files_exist = all(os.path.exists(f) for f in paths_to_check)

    if all_files_exist:
        print("Все необходимые файлы уже на месте. Загрузка пропущена.")
        return
    
    print("Необходимые файлы не найдены. Начинаю загрузку с Kaggle...")

    # 3. Скачиваем датасет
    try:
        # unzip=False - мы хотим сами контролировать процесс распаковки
        kaggle.api.dataset_download_files(dataset_slug, path=path, unzip=False)
        print(f"Файл '{zip_file_name}' успешно скачан.")
    except Exception as e:
        print(f"Произошла ошибка при скачивании данных: {e}")
        return # Прерываем выполнение, если не удалось скачать

    # 4. Распаковываем только нужные файлы из архива
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as z:
            for file_name in files_to_check:
                print(f"Распаковка файла '{file_name}'...")
                z.extract(file_name, path)
        print("Распаковка завершена.")
    except Exception as e:
        print(f"Произошла ошибка при распаковке архива: {e}")
        return

    # 5. Удаляем архив после распаковки, чтобы не занимать место
    finally:
        if os.path.exists(zip_file_path):
            os.remove(zip_file_path)
            print(f"Архив '{zip_file_name}' удален.")