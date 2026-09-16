# Anal Russia Klinik 2025

Пайплайн для поиска потенциально спорных упоминаний препаратов в российских клинических рекомендациях.

Проект объединяет словарный поиск, ручные фильтры, отчеты Aho-Corasick, подготовку кейсов для LLM и аудит ответов OpenRouter.

## Состояние результатов — сверено 16 сентября 2026

Последний найденный итоговый отчёт — от **5 мая 2026**, 4 121 блок и 5 688 оценок упоминаний. Полные метрики, состав моделей и ограничения оценки приведены в [главном README репозитория](../README.md). Источник актуальных чисел — `reports/llm/openrouter_all_results.json`; файл `.current.json` является устаревшим промежуточным снимком.

## Что делает проект

- Загружает клинические рекомендации из JSON.
- Загружает справочники препаратов, blacklist и ручные маркеры.
- Ищет совпадения с высоким recall через Aho-Corasick.
- Отделяет полезные совпадения от ложных совпадений через ручные фильтры.
- Группирует найденные позиции в LLM-блоки.
- Запускает LLM-проверку через OpenRouter или тестовый fake-провайдер.
- Хранит ручные фильтры, gold-разметку и полезные результаты OpenRouter в Git.

## Быстрый старт

Требования:

- Python 3.11 или новее.
- Docker Desktop, если нужен полный контейнерный прогон.
- OpenRouter API key, если нужен реальный LLM-прогон.

Установка зависимостей:

```powershell
py -m pip install -e ".[test]"
```

Быстрый тестовый прогон без внешнего LLM:

```powershell
py Main.py run --provider fake --output-dir reports/smoke
py -m pytest -q
```

Полный Aho-отчет:

```powershell
py Main.py aho-report --output reports/aho/host_words_by_search_word.json --workers 16
```

Docker-команды:

```powershell
docker compose build app
docker compose run --rm aho
docker compose run --rm g4f-smoke
```

## OpenRouter

Секреты хранятся локально и не коммитятся:

```powershell
Copy-Item config/openrouter.env.example config/openrouter.env
notepad config/openrouter.env
```

Ожидаемые переменные:

```text
OPENROUTER_API_KEY=...
OPENROUTER_MODEL=openai/gpt-5.4
```

Запуск gold-проверки:

```powershell
py reports\llm\run_openrouter_gold40.py --limit 40
```

Запуск полного OpenRouter-прогона:

```powershell
py reports\llm\run_openrouter_all.py --limit 100
```

## Ручные фильтры и результаты в Git

В репозиторий должны попадать:

- `config/manual_filters.csv`
- `data/samples/manual_filters.csv`
- `reports/aho/host_word_filters.json`
- `reports/aho/host_word_filters_parts/*.json`
- `reports/llm/excluded_preparations.json`
- `reports/llm/llm_gold_40*.json`
- `reports/llm/openrouter_gold40*.json`
- `reports/llm/openrouter_all_results*.json`

Причина: ручные фильтры, gold-разметка и результаты OpenRouter являются накопленной работой. Без Git такие файлы трудно восстановить.

В репозиторий не должны попадать:

- API-ключи и `*.env`.
- Логи dashboard/server.
- Python cache и pytest cache.
- Сырые отчеты больше лимита GitHub 100 MB.
- Временные partial JSONL.
- Локальная папка `old/`.

## Структура

- `Main.py` - локальная точка входа.
- `src/anal_russia_klinik/` - основной код.
- `src/bot_for_anal_rec/` - совместимость со старым именем пакета.
- `config/` - ручные фильтры и локальные настройки.
- `data/input/` - крупные входные JSON-файлы.
- `data/samples/` - маленькие фикстуры для тестов.
- `reports/aho/` - Aho-отчеты, dashboard wrappers, host-word фильтры.
- `reports/llm/` - LLM-кейсы, gold-разметка, OpenRouter-прогоны, dashboard wrappers.
- `docs/` - архитектура, операции, память агента, решения.
- `tests/` - pytest-проверки.

## Важные команды

```powershell
py Main.py --help
py Main.py run --provider fake --output-dir reports/smoke
py Main.py aho-report --output reports/aho/host_words_by_search_word.json --workers 16
py reports\aho\filter_detailed_host_words.py
py reports\aho\group_filtered_locations.py
py reports\llm\build_llm_review_cases.py --window-chars 2500
py reports\llm\run_openrouter_gold40.py --limit 40
py reports\llm\run_openrouter_all.py --limit 100
py -m pytest -q
```

## GitHub

Целевой репозиторий:

```text
https://github.com/TryDotAtwo/Anal_Russia_Klinik2025
```

Проверка remote:

```powershell
git remote -v
```

Если `origin` указывает на старый репозиторий:

```powershell
git remote set-url origin https://github.com/TryDotAtwo/Anal_Russia_Klinik2025.git
```

## Ограничения

GitHub отклоняет обычные файлы больше 100 MB. Поэтому крупные воспроизводимые отчеты не коммитятся напрямую. Для таких файлов нужен локальный пересчет, release artifact или Git LFS.

Текущие крупные локальные файлы:

- `reports/aho/host_words_by_search_word.json`
- `reports/llm/llm_review_cases.json`
- `data/input/clinical_recommendations.json`

## Память проекта

Долговременная память агента хранится в `docs/agent-memory.md`. При значимых изменениях нужно обновлять `docs/agent-memory.md`, `docs/operations-log.md` или профильный документ в `docs/`.

## Локальные производные артефакты: инфографика и CSV

CSV, PNG и `tools/build_openrouter_report_artifacts.py` имеются в локальной рабочей копии, но на 16 сентября 2026 отсутствуют в GitHub `main`. Команда пересборки ниже относится к этой локальной копии. Опубликованный источник итогов — `reports/llm/openrouter_all_results.json`.

Производный отчет построен без изменения исходного `reports/llm/openrouter_all_results.json`.

Локальные изображения: `reports/llm/infographic_slide_1.png`, `infographic_slide_2.png`, `infographic_slide_3.png`.

### Основные итоги накопленного отчёта

| Параметр | Значение |
|---|---:|
| Модель последнего запуска | `openai/gpt-5.4-mini` |
| Prompt version | `llm-review-v3-multi` |
| Блоки OpenRouter | 4 121 |
| Case-level оценки | 5 688 |
| Клинические рекомендации в CSV | 488 |
| Label accuracy на gold | 88.1% |
| Recommendation strength accuracy | 81.1% |
| Evidence level accuracy | 81.8% |

### Как читать оценку качества

Метрики рассчитаны на 143 вручную размеченных случаях: класс — 126/143, УУР — 116/143, УДД — 117/143. Совпадение отсутствующих УУР/УДД также засчитывается. Независимость этой выборки от настройки промптов не подтверждена.

Накопленный файл содержит 4 038 блоков с моделью запроса `openai/gpt-5.4-mini`, 80 — `openai/gpt-5.4`, 3 — `openai/gpt-4-turbo`. Возобновление сохраняет старые ответы даже при смене модели или промпта. Поэтому итоговые метрики нельзя приписывать только mini; старые gold-прогоны с другими выборками не являются прямым сравнением моделей.

`--limit` у `run_openrouter_all.py` обязателен и задаёт число новых вызовов. По умолчанию используется возобновление. Для реального прогона необходим локальный `reports/llm/llm_review_cases.json`, который не входит в Git из-за размера.

### Сколько найдено LLM-классов

| Класс | Количество |
|---|---:|
| Рекомендации | 3 088 |
| Противопоказания | 116 |
| Упоминания литературы | 686 |
| Ложные совпадения / ошибки | 1 794 |
| Неясно | 4 |

### Словарные источники: суммы столбцов локального CSV

| Источник | Количество совпадений |
|---|---:|
| MedIQ | 1 925 |
| Blacklist | 517 |
| Маркеры | 367 |

### CSV-таблица

Готовая таблица: `reports/llm/openrouter_all_results.csv`.

Формат: UTF-8 with BOM, разделитель `;`, строки агрегированы по ID клинической рекомендации. Колонки включают ID, название, МКБ-10, возраст, разработчика, дату размещения, статус, ссылку, уровни рекомендаций, счетчики MedIQ/Blacklist/маркеров и списки найденных терминов.

Уровни рекомендаций нормализуются перед выводом: кириллические `А/В/С` приводятся к латинским `A/B/C`, пустые `null`-части удаляются.

Повторная сборка:

```powershell
py tools\build_openrouter_report_artifacts.py
```
