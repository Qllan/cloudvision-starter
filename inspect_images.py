from pathlib import Path
from PIL import Image

folder = Path("data/input")

for file in sorted(folder.iterdir()):
    if file.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
        try:
            with Image.open(file) as image:
                print("Файл:", file.name)
                print("Размер:", image.size)
                print("Режим:", image.mode)
                print("Формат:", image.format)
                print("--------------------")
        except Exception as error:
            print("Ошибка:", file.name, error)
