# Anal Russia Klinik — запуск и воспроизведение

Актуальные результаты, качество LLM и состав моделей описаны в [главном README](../README.md). Последний сохранённый результат — 5 мая 2026; комплект для воспроизведения опубликован 16 сентября 2026.

## Установка и восстановление

Из корня репозитория:

```powershell
cd Bot_for_anal_rec
py -m pip install -e ".[test,reports]"
py tools/restore_snapshot.py
py tools/restore_snapshot.py --check
py -m pytest -q
```

Требуется Python 3.11+. В Linux/macOS замените `py` на `python3`. Для PNG нужен Arial или DejaVu Sans (Debian/Ubuntu: `fonts-dejavu-core`).

Восстановление не обращается к сети и не вызывает LLM. Три gzip-снимка в [data/snapshots](data/snapshots/manifest.json) восстанавливают:

- `data/input/clinical_recommendations.json` — корпус рекомендаций;
- `data/input/drugs.json` — справочник MedIQ;
- `reports/llm/llm_review_cases.json` — подготовленные кейсы и блоки.

Скрипт проверяет SHA-256; уже существующий файл с другим содержимым сохраняет и сообщает об ошибке. Распакованные файлы игнорируются Git. Архивы занимают около 59 МБ, распакованные данные — 549 МБ.

## Быстрый прогон без API

```powershell
py Main.py run --provider fake --texts data/samples/clinical.json --markers data/samples/markers.json --blacklist data/samples/blacklist.json --preparations data/samples/preparations.json --filter-file data/samples/manual_filters.csv --output-dir reports/smoke
```

Это проверка программы на маленьких примерах; она не измеряет качество реальной LLM.

## Готовые результаты и пересборка

- [Основной JSON](reports/llm/openrouter_all_results.json): 4 121 блок, 5 688 case-level ответов, метрики на 143 gold-случаях.
- [CSV](reports/llm/openrouter_all_results.csv): 488 документов, UTF-8 с BOM, разделитель `;`.
- [Экспертные страницы](reports/expert_review/index.html): 3 071 рекомендация и 115 противопоказаний после фильтра исключений.
- [Gold-страницы](reports/gold_review/index.html): 4 086 блоков.
- [Ручная разметка](reports/llm/llm_gold_40.json), [исключения](reports/llm/excluded_preparations.json), [фильтры](reports/aho/host_word_filters.json).

HTML открываются локально после клонирования; они содержат CSS/JS и данные. Изменения сохраняются в localStorage браузера. Экспортируйте JSON кнопкой на странице для резервного копирования и переноса разметки.

Пересборка из восстановленных данных без новых платных вызовов:

```powershell
py tools/build_openrouter_report_artifacts.py
py tools/build_expert_review_pages.py
py tools/build_gold_review_pages.py
```

Генераторы перезаписывают производные CSV, PNG и HTML. Основной JSON ответов и ручная разметка остаются исходными данными. Файл `openrouter_all_results.current.json` синхронизирован с итоговым; старый опубликованный промежуточный файл сохранён под именем `openrouter_all_results.checkpoint-20260504.json`.

В CSV счётчики MedIQ/Blacklist/маркеров считают уникальные названия внутри каждого документа. На инфографике источников считаются все совпадения внутри оценённых кейсов; эти показатели различаются.

## Новые OpenRouter-запросы

```powershell
Copy-Item config/openrouter.env.example config/openrouter.env
notepad config/openrouter.env
```

```text
OPENROUTER_API_KEY=...
OPENROUTER_MODEL=openai/gpt-5.4-mini
```

```powershell
py reports/llm/run_openrouter_all.py --limit 100
py reports/llm/run_openrouter_gold40.py --limit 40
```

`--limit` у полного runner обязателен и ограничивает новые вызовы. Возобновление по умолчанию сохраняет ответы предыдущих моделей и промптов. Итоговый файл смешанный: 4 038 блоков GPT-5.4-mini, 80 GPT-5.4 и 3 GPT-4-turbo по метаданным запросов. Поэтому общие метрики не являются отдельным сравнительным тестом mini.

Ключи и env-файлы локальны и не коммитятся. Новый платный прогон не нужен для воспроизведения сохранённых отчётов.

## Перестроение поиска и кейсов

Готовые отфильтрованные результаты уже включены. Для повторения самого раннего этапа после восстановления входных данных:

```powershell
py Main.py aho-report --output reports/aho/host_words_by_search_word.json --workers 16
py reports/aho/filter_detailed_host_words.py
py reports/aho/group_filtered_locations.py
py reports/llm/build_llm_review_cases.py --window-chars 2500
```

Полный сырой Aho-отчёт занимает около 2 ГБ, а частичные результаты требуют дополнительного места. Перестроение кейсов заменяет восстановленный снимок; для точного воспроизведения опубликованных итогов используйте исходный снимок.

Docker-вариант Aho после восстановления:

```powershell
docker compose build app
docker compose run --rm aho
```

Docker при публикации комплекта не проверялся.

## Структура и политика данных

- `Main.py` — точка входа; `src/anal_russia_klinik/` — основной пакет.
- `data/input/` — входные словари и корпус; `data/snapshots/` — сжатые снимки и SHA-256.
- `reports/aho/` — фильтры и результаты словарного поиска.
- `reports/llm/` — gold-разметка, исключения, ответы и отчёты.
- `reports/expert_review/`, `reports/gold_review/` — автономные страницы ручной проверки.
- `tools/` — восстановление снимков и генераторы; `tests/` — pytest.
- `docs/agent-memory.md` — память проекта; `docs/operations-log.md` — существенные изменения.

В Git сохраняются ручные фильтры и метки, исключения, полезные ответы и воспроизводимые снимки. Секреты, env-файлы, логи, кеши, временные файлы и сырой 2-ГБ Aho-отчёт исключены.
