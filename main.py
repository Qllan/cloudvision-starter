"""CloudVision AI: этап 1 — проверка исходных изображений.

На следующем этапе сюда будет подключена нейронная сеть.
Скрипт пока НЕ оценивает облачность и НЕ удаляет файлы.
"""

from pathlib import Path

from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_ROOT / "data" / "input"


def main() -> None:
    print("\n=== CloudVision AI: проверка изображений (этап 1) ===")
    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    paths = sorted(
        path for path in INPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        print("В папке data/input пока нет снимков.")
        print("Добавь туда 1–3 изображения JPG, PNG или TIFF и запусти снова.")
        return

    valid = 0
    for path in paths:
        try:
            with Image.open(path) as picture:
                width, height = picture.size
                picture.verify()
            print(f"✓ {path.name}: {width} × {height} пикселей")
            valid += 1
        except (UnidentifiedImageError, OSError, ValueError) as error:
            print(f"✗ {path.name}: не удалось прочитать файл ({error})")

    print(f"\nПрочитано: {valid} из {len(paths)} изображения(ий).")
    print("Следующий этап — обучение/подключение нейросети для облаков.")
    print("Пока проценты облачности НЕ вычисляются; файлы НЕ изменяются.")


if __name__ == "__main__":
    main()
