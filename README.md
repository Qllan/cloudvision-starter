# CloudVision AI 🛰️

Учебно-конкурсный проект для кейса **«СПУТНИКС»** на SK Kids Challenge 2026.

**Цель:** нейросеть определяет облачность спутниковых снимков в процентах. Если облачность **строго больше 75%**, изображение отбраковывается. В конце формируется отчёт с названием сохранённых снимков, процентами и названиями удалённых файлов. Итоговый материал — презентация `.pptx`.

> **Текущий статус: этап 1.** Реализованы открытие изображений и проверка правила 75%. Нейросети, анализа облачности и удаления файлов **пока нет**. Мы не выдаём простой просмотр файлов за работу ИИ.

## Что нужно

- Компьютер с Windows 10/11, macOS или Linux
- Python 3.11 или 3.12
- Visual Studio Code с расширением Python
- GitHub Desktop (необязательно, для удобной отправки проекта на GitHub)

## Шаг 1. Подготовь проект

Распакуй архив в удобную папку, например `Documents/CloudVisionAI`, и открой эту папку в VS Code (`File → Open Folder`).

## Шаг 2. Установи зависимости

Открой терминал VS Code (`Terminal → New Terminal`).

**Windows (PowerShell):**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Если Python 3.11 не установлен, используй `py -m venv .venv`. Если команда `py` не найдена, нужно установить Python.

**macOS/Linux:**

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

## Шаг 3. Проверь программу

**Windows:**

```powershell
.\.venv\Scripts\python.exe main.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

**macOS/Linux:**

```bash
./.venv/bin/python main.py
./.venv/bin/python -m unittest discover -s tests -v
```

Программа предупредит, что в `data/input` пока нет снимков. Это нормально.

## Шаг 4. Добавь изображения

Скопируй 1–3 спутниковых снимка `.png`, `.jpg` или `.tif` в папку `data/input`. Запусти `main.py` снова. Скрипт покажет имена файлов и их размеры, не изменяя оригиналы.

## Шаг 5. Опубликуй код на GitHub

Установи GitHub Desktop, выбери **File → Add Local Repository** и укажи распакованную папку. Если программа предложит создать Git-репозиторий здесь, согласись. Сделай первый коммит и нажми **Publish repository**. Поставь **Private**, если не хочешь пока публиковать промежуточную работу. Большие снимки и модели исключены через `.gitignore`.

## Что в репозитории

```text
CloudVisionAI/
├── main.py                # Этап 1: читаем снимки
├── cloudvision/
│   ├── __init__.py
│   └── rules.py           # Строго >75% = удаление (пока только правило)
├── tests/
│   └── test_rules.py     # Автоматические тесты условия 75%
├── data/
│   ├── input/             # Исходные снимки, вне Git
│   ├── accepted/          # Сохранённые (следующий этап)
│   └── rejected/          # Отбракованные (следующий этап)
├── docs/
│   └── project_plan.md
├── reports/               # Отчёты (следующий этап)
├── presentation/          # Будущая .pptx презентация
├── requirements.txt
└── README.md
```

## Официальное задание

https://kids.sk.ru/skkidschallenge/

Набор снимков, указанный в кейсе: https://disk.sputnix.ru/index.php/s/HbkKc34batcjkSX

Нейросеть будет добавлена и проверена на доступных спутниковых изображениях с реальными разметками облаков. Участнику важно уметь объяснить принцип работы и ограничения модели.
