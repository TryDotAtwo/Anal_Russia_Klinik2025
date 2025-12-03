import re
import asyncio
import aiohttp
import g4f
from Logic.Work_witch_word import generate_word_forms

async def find_mentions(drug, text, config, verbose=False):
    """Гибкий поиск упоминаний препарата в тексте с использованием словоформ."""

    drug_forms = generate_word_forms(drug)
    mentions = []
    text_lower = text.lower()
    pattern = re.compile(rf'(?:{"|".join(re.escape(form) for form in drug_forms)})(?:[а-яёa-z]*|\b)', re.IGNORECASE)
    if verbose:
        print(f"[{drug}] Поиск упоминаний с паттерном: {pattern.pattern}")
    for match in pattern.finditer(text_lower):
        start = max(0, match.start() - config["context_before"])
        end = min(len(text), match.end() + config["context_after"])
        context = text[start:end]
        mentions.append(context)
        if verbose:
            print(f"[{drug}] Найдено упоминание: {context[:50]}...")
    return mentions



async def analyze_mention(drug, context, session, max_retries=5, initial_delay=2):
    """
    Анализ упоминания препарата с помощью LLM через g4f.
    Основная модель: command_r, при ошибке 429 автоматически переключаемся на deepseek_r1.
    Ответ ожидается строго в виде JSON-объекта.
    """
    prompt = f"""
Проанализируй фрагмент клинической рекомендации.

Контекст:
\"\"\"
{context}
\"\"\"

Препарат: "{drug}".

ВНИМАНИЕ: препарат может быть упомянут случайно (например, как микроэлемент «селен», а не БАД «Селен актив»).

ВНИМАНИЕ — БЫВАЮТ ЛОЖНЫЕ СРАБАТЫВАНИЯ:
- «селен» — это микроэлемент, а не БАД «Селен актив»
- «АТФ» — это молекула энергии, а не препарат
- «омега» — жирные кислоты, а не «Омега-3» как БАД
- «витамин» в тексте ≠ «Поливитамины» как препарат

Твоя задача:
1. Проверить — действительно ли в этом фрагменте идёт речь именно о препарате/БАДе под названием «{drug}» как о лекарственном средстве, методе или биодобавке.
2. Если это ложное срабатывание (например, «селен» как химический элемент, «АТФ» в биохимическом смысле и т.д.) — обязательно укажи тип "Ошибочное".


Тебе нужно:
1. Найти, как именно в этом фрагменте упоминается препарат "{drug}".
2. Определить (если возможно):
   - УДД (уровень достоверности доказательств, обычно цифра 1–5, иногда не указан).
   - УУР (уровень убедительности рекомендаций, обычно буква A/B/C/D или аналог, иногда не указан).
   - Тип упоминания: одно из значений:
     - "рекомендация"
     - "литература"
     - "противопоказание"
     - "Ошибочное" (если на самом деле препарат не упоминается по смыслу)
3. Дать короткий комментарий (1–2 предложения), почему ты так решил(а).

Требования к формату ответа:
- Ответ ДОЛЖЕН быть ОДНИМ JSON-объектом без пояснений, текста до или после него.
- Никаких Markdown, никаких ```json и ```.
- Если значение не найдено, пиши строку "NULL".

Контекст:
\"\"\"
{context}
\"\"\"

Препарат: "{drug}".


Структура JSON (ключи строго такие):
{{
  "УДД": "<строка или \"NULL\">",
  "УУР": "<строка или \"NULL\">",
  "Тип": "<\"рекомендация\"|\"литература\"|\"противопоказание\"|\"Ошибочное\"|\"NULL\">",
  "Комментарий": "<короткий текст или \"NULL\">"
}}

Ещё раз: верни ТОЛЬКО этот JSON-объект, без форматирования Markdown и без других полей.


"""

    # Цепочка моделей для автоматического переключения при rate limit (429)
    models_chain = [
        g4f.models.command_r,
        g4f.models.deepseek_r1,
        g4f.models.llama_3_70b,
        g4f.models.command_r_plus,
        g4f.models.command_a,
        g4f.models.qwen_3_30b,
        g4f.models.qwen_3_14b
    ]
    current_model_idx = 0

    for attempt in range(max_retries):
        model = models_chain[current_model_idx]
        try:
            response = await asyncio.to_thread(
                g4f.ChatCompletion.create,
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
            )
            return await parse_gpt_response(response)
        except Exception as e:
            err_text = str(e)
            model_name = getattr(model, "name", str(model))
            print(
                f"Ошибка анализа упоминания {drug} (модель: {model_name}, "
                f"попытка {attempt + 1}/{max_retries}): {err_text}"
            )

            # Если получили rate limit (429), пробуем переключиться на следующую модель
            if "429" in err_text or "rate limit" in err_text.lower():
                if current_model_idx + 1 < len(models_chain):
                    current_model_idx += 1
                    next_model = models_chain[current_model_idx]
                    next_name = getattr(next_model, "name", str(next_model))
                    print(f"Переключаемся на резервную модель {next_name}")

            if attempt < max_retries - 1:
                delay = initial_delay * (attempt + 1)  # лёгкий рост задержки
                print(f"Повторная попытка через {delay} секунд...")
                await asyncio.sleep(delay)
            else:
                print(f"Не удалось обработать упоминание {drug} после {max_retries} попыток.")
                return {
                    "УДД": "Ошибка",
                    "УУР": "Ошибка",
                    "Тип": "Ошибка",
                    "Комментарий": None,
                    "recommended": False,
                    "contraindicated": False,
                    "error": err_text,
                }

async def parse_gpt_response(response):
    """Парсинг и валидация JSON-ответа GPT."""
    raw_content = response.get("content", "") if isinstance(response, dict) else str(response)

    # Убираем возможные Markdown-обёртки ```json ... ``` или ``` ... ```
    content = raw_content.strip()
    # Вырезаем первый JSON-объект по фигурным скобкам
    json_candidate = None
    match = re.search(r'\{.*\}', content, re.DOTALL)
    if match:
        json_candidate = match.group(0)
    else:
        json_candidate = content

    try:
        import json as _json
        parsed = _json.loads(json_candidate)
    except Exception:
        # Попытка 2: иногда модель может вернуть список с одним объектом
        try:
            parsed = _json.loads(f"[{json_candidate}]")[0]
        except Exception:
            # Фолбэк: полностью дефолтный результат
            return {
                "УДД": "Не определен",
                "УУР": "Не определен",
                "Тип": "Не определен",
                "Комментарий": None,
                "recommended": False,
                "contraindicated": False,
                "error": "json_parse_error",
            }

    # Приводим к ожидаемой структуре
    if isinstance(parsed, list) and parsed:
        obj = parsed[0]
    elif isinstance(parsed, dict):
        obj = parsed
    else:
        obj = {}

    def norm_field(name):
        val = obj.get(name)
        if val is None:
            return "Не определен"
        if isinstance(val, str):
            val_stripped = val.strip()
            if not val_stripped or val_stripped.upper() == "NULL":
                return "Не определен"
            return val_stripped
        return str(val)

    udd = norm_field("УДД")
    uur = norm_field("УУР")
    t = norm_field("Тип")
    comment_raw = obj.get("Комментарий")
    if isinstance(comment_raw, str):
        comment = comment_raw.strip()
        if not comment or comment.upper() == "NULL":
            comment = None
    else:
        comment = None

    t_lower = t.lower()
    recommended = "рекомендация" in t_lower
    contraindicated = "противопоказание" in t_lower

    return {
        "УДД": udd,
        "УУР": uur,
        "Тип": t,
        "Комментарий": comment,
        "recommended": recommended,
        "contraindicated": contraindicated,
    }
