# Anal Russia Klinik 2025

Пайплайн для поиска спорных упоминаний препаратов в российских клинических рекомендациях.

Рабочий код находится в каталоге `Bot_for_anal_rec/`.

## Быстрый старт

```powershell
cd Bot_for_anal_rec
py -m pip install -e ".[test]"
py Main.py run --provider fake --output-dir reports/smoke
py -m pytest -q
```

## Основная документация

Полное описание проекта: `Bot_for_anal_rec/README.md`.

## Целевой GitHub-репозиторий

```text
https://github.com/TryDotAtwo/Anal_Russia_Klinik2025
```

Если `origin` указывает на старый репозиторий:

```powershell
git remote set-url origin https://github.com/TryDotAtwo/Anal_Russia_Klinik2025.git
```

## Git policy

В Git хранятся исходный код, документация, тесты, ручные фильтры, gold-разметка и полезные OpenRouter-результаты.

В Git не хранятся секреты, кэши, логи, временные partial-файлы и обычные файлы больше лимита GitHub 100 MB.
