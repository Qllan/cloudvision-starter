# CloudVision AI — СПУТНИКС 🛰️

**Конкурсный прототип сегментации облаков на спутниковых снимках** (Python, PyTorch, компактная U-Net, CUDA, SQLite).

**Задача:** оценить облачность каждого кадра и предложить **ОТСЕЯТЬ**, только если облака занимают **строго больше 75%** площади. Ровно 75% → **СОХРАНИТЬ**.

> ⚠️ Модель остаётся исследовательской. Она ошибается на тонкой облачности и рельефе. Решения — рекомендации оператору; **исходные фотографии автоматически не удаляем**.

## Что показать жюри

1. Оригинальный спутниковый снимок и розовая маска облаков.
2. Вычисленный процент и предложение «СОХРАНИТЬ / ОТСЕЯТЬ».
3. Пакетный запуск на CUDA с восстановлением прогресса после прерывания.
4. Ручную проверку и ограничения на независимых данных.

### Зафиксированные результаты

| Проверка | v4 Policy | v5 Pilot | Что важно |
|---|---:|---:|---|
| VAL, 10 снимков | MAE 3,455 п.п. | MAE 3,316 п.п. | Использовалась для выбора эпох |
| Независимый TEST, 24 снимка | MAE 16,63 п.п.; 23/24 решений | MAE 16,43 п.п.; 23/24 | **0** снимков с эталоном >75% |
| Целевой *cloud-heavy challenge*, 13 сцен | 3 из 7 настоящих отсевов | 3 из 7 | Специально выбранные сложные кадры, **не** случайная выборка |

В целевом challenge обе модели **пропустили 4 из 7** отсеваемых кадров. Это причина обязательного ручного контроля. Предыдущее сравнение v3/v4 на пяти TEST также сохранено отдельно. Не смешивать метрики разных выборок.

## Быстрый запуск на Windows

В корне репозитория:

~~~powershell
# Проверить, доступна ли видеокарта:
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"

# Создать безопасный пилот на CUDA. Фото не перемещаются и не удаляются:
.\.venv\Scripts\python.exe run_batch_v4.py --input data/new_raw --model models/cloudvision_unet_v4_policy.pt --out reports/demo_for_jury --device cuda --limit 30

# Создать операторскую HTML-галерею с наложениями:
.\.venv\Scripts\python.exe build_operator_report_v4.py --input data/new_raw --batch reports/demo_for_jury
Start-Process .\reports\demo_for_jury\operator_review\index.html
~~~

Для уже размеченного пилота используй \`--verified data/pilot_verified_v4\` в \`build_operator_report_v4.py\`. Скрипт не выдаёт AI-маски за подтверждённые человеком.

### Диагностика самого сложного cloud-heavy кадра

~~~powershell
.\.venv\Scripts\python.exe compare_cloud_heavy_models.py --device cuda
Start-Process .\reports\cloud_heavy_case_review\2025-11-30_22-18-23_SXC3-2334_1_v4_v5_comparison.jpg
~~~

Две замороженные модели и список TEST сверяются по SHA-256, а оригиналы и исходные оценки остаются неизменными.

## Основные файлы

| Задача | Скрипт |
|---|---|
| Безопасная пакетная обработка, SQLite и повторный запуск | \`run_batch_v4.py\` |
| Загрузка весов и инференс (без операций удаления) | \`cloudvision/inference.py\` |
| Операторский HTML/CSV-отчёт | \`build_operator_report_v4.py\` |
| Редактор ручных масок | \`label_with_ai_fast.py\` |
| Отчёт для защиты из зафиксированных CSV | \`build_competition_report.py\` |
| Сравнение замороженных v4/v5 на одном снимке | \`compare_cloud_heavy_models.py\` |
| Обучение v4/v5 (для воспроизведения экспериментов) | \`train_cloudvision_v4.py\`, \`train_cloudvision_v5.py\` |

Сохранены и вспомогательные исторические скрипты, которые нужны для воспроизведения экспериментов. Промежуточные \`README_STAGE*\` удалены из актуального дерева репозитория: они доступны в истории Git.

## Документация

- [Конкурсный HTML-отчёт](docs/competition_report.md)
- [Обработка с CUDA](docs/gpu_batch_v4.md)
- [Операторская проверка](docs/operator_review_v4.md)
- [Правила независимого TEST](docs/final_holdout_v4.md)
- [Проверка сложной облачности](docs/cloud_heavy_challenge.md)
- [Обучение v5](docs/v5_training.md)
- [Обслуживание репозитория](docs/repository_maintenance.md)

## Тесты и безопасность

~~~powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_batch_v4.py" -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_inference_boundary.py" -v
git status --short
~~~

CI запускает проверки без обучения моделей на **Windows и Linux**.

\`data/\`, \`models/\`, \`reports/\`, \`checkpoints/\`, локальные маски, оригинальные фото и конкурсные итоговые файлы **не добавлять в Git**. Проект находится в ветке \`feature/gpu-batch-resume\`; \`main\` не объединять без отдельного решения.
