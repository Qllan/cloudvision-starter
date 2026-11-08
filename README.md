# CloudVision AI 🛰️

Проект по кейсу компании **СПУТНИКС** для SK Kids Challenge: анализ облачности
спутниковых изображений с помощью нейросети, расчёт процента облачности,
сортировка по правилу **строго более 75% — отсев**, отчёт оператору.

> Исследовательский прототип, не проверенная промышленная система. Модель иногда
> ошибается, поэтому автоматическое удаление оригинальных спутниковых снимков запрещено.

## Что уже работает

- Обучение компактной U-Net (train_cloudvision_v3.py) и сравнение Area / Policy.
- Предпросмотр результатов сортировки (run_competition_demo.py).
- Быстрый редактор ручной разметки (label_with_ai_fast.py).
- Эксперименты и тесты предыдущих этапов.

## Новое в ветке v4: GPU + пакетная обработка

run_batch_v4.py обрабатывает **новые снимки по одному**, использует CUDA при
наличии совместимого PyTorch, рекурсивно находит изображения, сохраняет
маски и журнал прогресса (SQLite). Повторный запуск пропускает готовые маски,
если исходный файл и модель не изменились. **Файлы не удаляются и не копируются.**

### Windows 11 — первые пять снимков

Понадобятся установленный Python, PyTorch с CUDA (в совместимом окружении),
Pillow, NumPy и веса models/cloudvision_unet_v3_policy.pt.
Инструкции по PyTorch: https://pytorch.org/get-started/locally/

    .\.venv\Scripts\python.exe -c "import torch; print('CUDA:', torch.cuda.is_available())"
    .\.venv\Scripts\python.exe run_batch_v4.py --input data/new_raw --out reports/batch_v4_new --device cuda --limit 5

Чтобы продолжить и обработать остальные снимки, запусти ту же команду без
--limit 5. Если прервёшь запуск, последующий старт продолжит работу.
Используй **отдельную папку --out для каждого исходного датасета**.

    .\.venv\Scripts\python.exe run_batch_v4.py --input data/new_raw --out reports/batch_v4_new --device cuda

Результаты:
- reports/batch_v4_new/operator_preview.csv — прогноз облачности и решение;
- reports/batch_v4_new/masks/ — бинарные маски по исходным подпапкам;
- reports/batch_v4_new/progress.sqlite3 — прогресс после каждого кадра.

Для исправления предсказаний в быстром редакторе:

    .\.venv\Scripts\python.exe label_with_ai_fast.py --input data/new_raw --ai-masks reports/batch_v4_new/masks --out data/verified_new

Не загружай изображения, разметку и контрольные точки модели в GitHub.

## Улучшение качества на новом датасете СПУТНИКС (v4)

После проверки пяти RGB-снимков текущая U-Net v3 показала заметные ложные
срабатывания на море и поверхности. Для улучшения качества сначала выбираем
50 **разнообразных** снимков, а не обрабатываем все 4207 без проверки.

    .\.venv\Scripts\python.exe select_v4_samples.py --input data/new_raw --count 50 --out reports/selection_v4/selection.csv

Маски строим только для TRAIN и VAL. Новую независимую TEST-группу не
используем для выбора модели:

    .\.venv\Scripts\python.exe run_batch_v4.py --input data/new_raw --out reports/batch_v4_selected --manifest reports/selection_v4/selection.csv --splits train,val --device cuda

Проверяем и исправляем маски в редакторе:

    .\.venv\Scripts\python.exe label_with_ai_fast.py --input data/new_raw --ai-masks reports/batch_v4_selected/masks --out data/verified_v4 --manifest reports/selection_v4/selection.csv --splits train,val

Подробнее: [отбор и разметка 50 кадров](docs/label_v4_samples.md).

## Подробная документация

- [Обработка больших датасетов v4](docs/gpu_batch_v4.md)
- [Быстрый редактор масок](docs/fast_labeler.md)
- [Условия и цели проекта](docs/project_plan.md)
- [Этапы экспериментов](README_STAGE14.md)

## Проверка тестов пакетного движка

    .\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_batch_v4.py" -v
