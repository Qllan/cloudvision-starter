# CloudVision AI — этап 14: конкурсная демонстрация

Этот этап **не переобучает** нейросеть. Он загружает сохранённую модель `models/cloudvision_unet_v3_policy.pt` и последовательно анализирует JPEG в `data/input`.

> **Ограничение:** модель Policy верно классифицировала 15 из 16 тестовых снимков, но пропустила 1 из 2 снимков с облачностью >75%. Это прототип, а не надёжная система для автоматического удаления оригиналов.

## Установка (Windows 11)

Скопируй `run_competition_demo.py` и `prepare_demo_copies.py` в корень проекта рядом с `train_cloudvision.py`. `test_stage14.py` положи в `tests`.

Используются уже установленные `torch`, `numpy` и `Pillow`. Нужны сохранённые файлы модели `models/cloudvision_unet_v3_policy.pt` и исходный модуль `train_cloudvision.py` из этапа 10.

### 1. Предпросмотр первых пяти снимков (никаких изменений оригиналов)

```powershell
.\.venv\Scripts\python.exe run_competition_demo.py --limit 5
```

Посмотри `reports/competition_demo/operator_report.csv`, `plan.csv`, `masks`, `previews`.

### 2. Вся папка снимков, с дополнительным сравнением с Area

```powershell
.\.venv\Scripts\python.exe run_competition_demo.py --area-model models/cloudvision_unet_v3_area.pt
```

`нужна_проверка=ДА` означает, что прогноз в пределах 15 п.п. от порога 75% или Policy и Area расходятся больше чем на 20 п.п. Это **предупреждение**, не исправленный прогноз.

### 3. Режим сохранения подходящих снимков

```powershell
.\.venv\Scripts\python.exe run_competition_demo.py --mode copy
```

Копии подходящих снимков появятся в `data/accepted_demo`, оригиналы останутся в `data/input`. При повторном запуске тот же файл нельзя перезаписать: об этом появится ошибка.

### 4. ДЕМОНСТРАЦИЯ удаления только рабочих копий

**Осторожно:** последняя команда действительно удаляет отклонённые копии из `data/work_input`. Никогда не перемещай туда единственные оригиналы. До демонстрации сделай резервную копию. Если не нужно показывать удаление, этот шаг можно не выполнять.

```powershell
.\.venv\Scripts\python.exe prepare_demo_copies.py --limit 5
.\.venv\Scripts\python.exe run_competition_demo.py --input data/work_input --mode preview --out reports/competition_demo_work
.\.venv\Scripts\python.exe run_competition_demo.py --input data/work_input --mode delete --confirm-delete-copies --out reports/competition_demo_work
```

Если среди 5 снимков нет предсказаний **выше 75%**, удалений не будет — это нормальный результат. Для честной демонстрации найди подходящие **исходные спутниковые снимки** и повтори подготовку копий; не подделывай оценки или снимки.

### Что сохраняется в отчёте

- Исходное имя файла и прогноз облачности в процентах;
- Вердикт `СОХРАНИТЬ` при значении <=75%, `ОТСЕЯТЬ` при >75%;
- Фактическое действие (просмотр, копирование, удаление рабочей копии);
- Имя сохранённого / удалённого файла, если применимо;
- Маска обнаруженных облаков и изображение предпросмотра;
- Предупреждения о спорных прогнозах.

Все результаты будут в `reports/competition_demo/`. Модели и оригиналы снимков программа не изменяет. Весь этап можно показать жюри, но в презентации необходимо честно указать ограничения точности.

## Тесты

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## GitHub

Загрузи в GitHub только новые `.py`, тест и README. Большие JPEG, контрольные точки `.pt`, личные пути, маски и результаты в репозиторий загружать не нужно. Убедись, что `.gitignore` покрывает `data/`, `models/`, `reports/`, `.venv/` и `__pycache__/`.
