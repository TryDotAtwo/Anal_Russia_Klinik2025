# Входные данные

Маркеры `AXTUNG.Json`, blacklist и `MetaData.json` включены напрямую.
Корпус `clinical_recommendations.json` и MedIQ `drugs.json` хранятся сжатыми в `../snapshots/`.
Из каталога проекта выполните `py tools/restore_snapshot.py`, затем `py tools/restore_snapshot.py --check`.
Эта же команда восстанавливает `reports/llm/llm_review_cases.json`. Контрольные суммы находятся в `data/snapshots/manifest.json`.

Это снимок данных для сохранённого майского отчёта 2026 года, а не автоматически обновляемый реестр.
