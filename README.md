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

## Обучение U-Net v4 на RTX 3060 Ti

После завершения ручной разметки 45 TRAIN/VAL снимков (35 TRAIN, 10 VAL)
можно дообучить модель v3 Policy на проверенных масках. Дополнительно
используется только старый TRAIN; старые и новые TEST не открываются.

Сначала короткая проверка в отдельной модели:

    .\.venv\Scripts\python.exe train_cloudvision_v4.py --device cuda --epochs 2 --model models/cloudvision_unet_v4_smoke.pt --out reports/training_v4_smoke

После её успешного завершения:

    .\.venv\Scripts\python.exe train_cloudvision_v4.py --device cuda --epochs 25 --size 256 --batch-size 4

Новый checkpoint: models/cloudvision_unet_v4_policy.pt.
Сравнение v3 и v4 на десяти проверочных снимках: reports/training_v4/model_comparison.csv.
Старый checkpoint v3 остаётся неизменным. Если старое разбиение
отсутствует, допустим отдельный эксперимент с --no-old-train.

Если v4 уже обучена и нужно сравнить **две независимые цели выбора эпохи**
(MAE процента облачности и IoU маски), запускай отдельный эксперимент без
перезаписи старого checkpoint:

    .\.venv\Scripts\python.exe train_cloudvision_v4.py --device cuda --epochs 25 --size 256 --batch-size 4 --model models/cloudvision_unet_v4_dual_area.pt --mask-model models/cloudvision_unet_v4_dual_mask.pt --out reports/training_v4_dual

Итог: три строки сравнения v3 / v4-area / v4-mask в
reports/training_v4_dual/model_comparison.csv. Выбор эпох делается по VAL,
TEST остаётся отдельно.

Подробнее: [дообучение v4 и проверка на VAL](docs/train_v4.md).

## Независимый финальный TEST (5 снимков)

Веса v4 уже выбраны по VAL. Чтобы честно проверить качество на пяти
отложенных TEST, **сначала зафиксируй модели**, затем вручную разметь
TEST без AI-масок, и только потом сравни их результаты.

    .\.venv\Scripts\python.exe evaluate_v4_holdout.py --freeze

    .\.venv\Scripts\python.exe label_with_ai_fast.py --input data/new_raw --manifest reports/selection_v4/selection.csv --splits test --blind --out data/test_verified_v4

После сохранения пяти новых эталонных масок:

    .\.venv\Scripts\python.exe evaluate_v4_holdout.py --evaluate --device cuda

Отчёт: reports/final_test_v4/summary.csv и per_image.csv.
Старые и новые исходные снимки, TRAIN/VAL-маски и модели не изменяются.
Подробная инструкция: [слепой TEST и фиксация моделей](docs/final_holdout_v4.md).

## Операторская галерея для пилота (30 снимков)

После безопасного запуска \`run_batch_v4.py\` с выводом в
\`reports/competition_pilot_v4\` можно создать отчёт без повторного
инференса:

    .\.venv\Scripts\python.exe build_operator_report_v4.py --input data/new_raw --batch reports/competition_pilot_v4
    Start-Process .\reports\competition_pilot_v4\operator_review\index.html

HTML показывает RGB и наложение маски ИИ; CSV выделяет снимки возле
конкурсного порога, почти 0% и потенциальные отсеваемые кадры.
Ни один оригинал не удаляется. Эти предупреждения не гарантируют, что
непомеченные кадры распознаны правильно.


Для исправления ложных облаков на суше/воде используй **операторскую
очередь** — она включает все 30 пилотных кадров, начиная с приоритетных:

    .\.venv\Scripts\python.exe label_with_ai_fast.py --input data/new_raw --ai-masks reports/competition_pilot_v4/masks --out data/pilot_verified_v4 --review-csv reports/competition_pilot_v4/operator_review/operator_review.csv

После сохранения вручную проверенных масок пересоздай отчёт:

    .\.venv\Scripts\python.exe build_operator_report_v4.py --input data/new_raw --batch reports/competition_pilot_v4 --verified data/pilot_verified_v4

В HTML теперь отдельно отображаются **решение нейросети** и **статус
ручной проверки**; отсутствие автоматических флагов не означает
проверенность. CSV содержит расхождение с вручную проверенной маской
для тех кадров, где она сохранена. Старые TRAIN/VAL/TEST не изменяются.

Подробнее: [операторский отчёт v4](docs/operator_review_v4.md).

## Обучение v5 с учётом 30 проверенных пилотных снимков

После пилотного аудита U-Net v4 (30 ручных масок, 3 пропущенных отсева)
подготовлено дообучение на исправленных TRAIN с отдельным новым TEST.
Оно **не использует** прежние TEST для подбора весов.

Сначала проверить пересечения с прежними TRAIN/VAL/TEST и создать
фиксированный набор из 24 ещё не использованных групп сцен:

    .\.venv\Scripts\python.exe prepare_v5_plan.py
    .\.venv\Scripts\python.exe select_v5_holdout.py --count 24

Только если обе команды завершились без ошибок — проверить CUDA:

    .\.venv\Scripts\python.exe train_cloudvision_v5.py --device cuda --epochs 2 --model models/cloudvision_unet_v5_smoke.pt --out reports/training_v5_smoke

После пробного обучения отдельный запуск v5:

    .\.venv\Scripts\python.exe train_cloudvision_v5.py --device cuda --epochs 18 --size 256 --batch-size 4

Новая модель сохраняется в \`models/cloudvision_unet_v5_pilot.pt\`,
старые v3 и v4 неизменны. Прежняя v4 VAL используется только для выбора
эпохи; новый \`reports/v5/holdout_manifest.csv\` **не открывается как
изображения** во время тренировки. После фиксации v5 новый TEST
размечается независимо, без подсказки AI.

Подробнее: [аудит, обучение и независимая проверка v5](docs/v5_training.md).

## Итоговый конкурсный отчёт: v3 / v4 / v5

После завершения слепой проверки 24 новых снимков собран самостоятельный
отчёт для защиты. Он сравнивает два независимых TEST (5 и 24 снимка),
10 VAL и операторский пилот из 30 кадров, показывает типичные ошибки,
график обучения и ограничения правила >75%.

    .\.venv\Scripts\python.exe build_competition_report.py
    Start-Process .\reports\competition_final\cloudvision_report.html

Отчёт офлайн и **ничего не удаляет**. Для печати в браузере:
**Печать → Сохранить как PDF**. Если готовый HTML уже существует,
скрипт не заменяет его; укажи новое \`--out\` для следующей версии.

**Вывод из независимого TEST 24:** MAE v4=16.63 п.п., v5=16.43 п.п.;
IoU v4=0.48290, v5=0.48222; обе модели — 23/24 правильных действий.
На этом TEST не было эталонов с облачностью >75%, поэтому надёжность
отсева таких кадров **не подтверждена**. Исторические TEST не
используем для дальнейшей настройки checkpoint.

Подробнее: [отчёт и конкурсная демонстрация](docs/competition_report.md).

## Отдельная проверка сильно облачных сцен (>75%)

Два независимых TEST (5 и 24 фото) не содержали эталонных кадров с
облачностью >75%. Для проверки правила отсева предусмотрен отдельный
**targeted challenge**, не случайный TEST:

    .\.venv\Scripts\python.exe prepare_cloud_heavy_challenge.py browse --count 100
    Start-Process .\reports\v5\cloud_heavy_pool\index.html

В галерее **только RGB**, без предсказаний ИИ. Отметь 10–15 визуально
сильно облачных фото из разных групп дата/камера, скачай
\`selected_cloud_heavy.txt\` в корень проекта и выполни:

    .\.venv\Scripts\python.exe prepare_cloud_heavy_challenge.py freeze

После этого **заморозь обе уже обученные модели до ручной разметки**.
Полная безопасная инструкция, включая \`--blind\` и однократную
оценку: [проверка сильно облачных сцен](docs/cloud_heavy_challenge.md).

Целевой отбор по виду облаков не позволяет делать выводы о средней
точности для всех 4207 снимков. Не удаляй исходники автоматически.

## Подробная документация

- [Обработка больших датасетов v4](docs/gpu_batch_v4.md)
- [Быстрый редактор масок](docs/fast_labeler.md)
- [Условия и цели проекта](docs/project_plan.md)
- [Этапы экспериментов](README_STAGE14.md)

## Проверка тестов пакетного движка

    .\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_batch_v4.py" -v
