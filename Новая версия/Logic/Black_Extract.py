import re
import ahocorasick

def get_word_stem(word):
    """Определяет основу слова, отрезая типичные окончания."""
    word = word.lower().strip()
    endings = [
        "ая", "яя", "ия", "ое", "ее", "ье", "ый", "ий", "ой", "ые", "ие",
        "ого", "ому", "ым", "ими", "ую", "ей", "им", "их", "ыми", "ою",
        "а", "я", "о", "е", "ь", "й", "и", "ы", "у", "ю", "ом", "ем", "ём",
        "ами", "ями", "ах", "ях", "ов", "ев", "ёв", "ам", "ям", "ей", "ой", "ою", "ми", "мя", "ин", "ын",
        "ть", "ти", "чь", "у", "ю", "ешь", "ет", "ем", "ете", "ут", "ют",
        "л", "ла", "ло", "ли", "я", "а", "в", "вши", "ши",
        "ущий", "ющий", "ащий", "ящий", "вший", "ший", "емый", "омый", "имый"
    ]
    for ending in sorted(endings, key=len, reverse=True):
        if word.endswith(ending):
            return word[:-len(ending)]
    return word

def generate_word_forms(word):
    """Генерирует все возможные словоформы слова для всех частей речи и падежей."""
    word = word.lower().strip()
    stem = get_word_stem(word)
    noun_endings = [
        "", "-а", "-я", "-о", "-е", "-ь", "-й", "-и", "-ы", "-у", "-ю", "-ом", "-ем", "-ём",
        "-ами", "-ями", "-ах", "-ях", "-ов", "-ев", "-ёв", "-ам", "-ям", "-ей", "-ой", "-ою",
        "-ми", "-мя", "-ин", "-ын"
    ]
    adj_endings = [
        "-ый", "-ий", "-ой", "-ая", "-яя", "-ия", "-ое", "-ее", "-ье", "-ые", "-ие",
        "-ого", "-ому", "-ым", "-ими", "-ую", "-ей", "-им", "-их", "-ыми", "-ой", "-ою"
    ]
    verb_endings = [
        "-ть", "-ти", "-чь", "-у", "-ю", "-ешь", "-ет", "-ем", "-ете", "-ут", "-ют",
        "-л", "-ла", "-ло", "-ли", "-я", "-а", "-в", "-вши", "-ши",
        "-ущий", "-ющий", "-ащий", "-ящий", "-вший", "-ший", "-емый", "-омый", "-имый"
    ]
    adverb_endings = ["-о", "-е", "-и", "-у"]
    forms = [stem + ending for ending in noun_endings + adj_endings + verb_endings + adverb_endings]
    forms.append(word)
    return list(set(forms))

def create_drug_dict(blacklist_drugs):
    """Преобразование списка препаратов в словарь для быстрого доступа.
       Ключ — название препарата (и альтернативные варианты), значение — описание."""
    drug_dict = {}
    for entry in blacklist_drugs:
        main_name = entry["Название препарата"].lower().strip()
        drug_dict[main_name] = entry["Описание"]
        for alt in entry["Альтернативные названия"]:
            alt_name = alt.lower().strip()
            drug_dict[alt_name] = entry["Описание"]
    return drug_dict

def build_aho_automaton_word_forms(drug_dict):
    """
    Строит Ахо–Корасик автомат для всех словоформ, сгенерированных по ключам drug_dict.
    В результате, каждый ключ автомата — это словоформа, а значение — список оригинальных ключей,
    для которых эта форма была получена.
    """
    form_to_keys = {}
    for key in drug_dict.keys():
        key_clean = key.lower().strip()
        forms = generate_word_forms(key_clean)
        for form in forms:
            form_clean = form.lower().strip()
            if form_clean not in form_to_keys:
                form_to_keys[form_clean] = [key_clean]
            else:
                if key_clean not in form_to_keys[form_clean]:
                    form_to_keys[form_clean].append(key_clean)
    A = ahocorasick.Automaton()
    for form, keys in form_to_keys.items():
        A.add_word(form, keys)
    A.make_automaton()
    return A

def match_exact(automaton, s):
    """
    Ищет точное совпадение строки s в автомате.
    Если найдено, возвращает значение (список оригинальных ключей), иначе – None.
    """
    try:
        return automaton.get(s)
    except KeyError:
        return None

def is_in_list_aho(drug_name, automaton):
    """
    Проверка наличия препарата в расстрельном списке с учётом окончаний,
    используя заранее построенный Ахо–Корасик автомат.
    """
    drug_name = drug_name.lower().strip()
    forms = generate_word_forms(drug_name)
    for form in forms:
        if match_exact(automaton, form) is not None:
            return True
    return False

def get_description_aho(drug_name, drug_dict, automaton, visited=None):
    """
    Получение описания препарата с учётом ссылок 'см.' с использованием Ахо–Корасик автомата.
    Если описание начинается с "см.", производится попытка перейти по ссылке.
    """
    if visited is None:
        visited = set()
    drug_name = drug_name.lower().strip()
    if drug_name in visited:
        return None
    visited.add(drug_name)
    forms = generate_word_forms(drug_name)
    for form in forms:
        matched_keys = match_exact(automaton, form)
        if matched_keys:
            # Берём первый оригинальный ключ из списка
            orig_key = matched_keys[0] if isinstance(matched_keys, list) else matched_keys
            desc = drug_dict[orig_key]
            cleaned_desc = desc.strip()
            if cleaned_desc.lower().startswith("см."):
                match = re.search(r"см\.\s*([^.,]+)", desc, re.IGNORECASE)
                if match:
                    ref_text = match.group(1).strip()
                    word_count = len(ref_text.split())
                    if word_count <= 3:
                        ref_desc = get_description_aho(ref_text, drug_dict, automaton, visited)
                        return ref_desc if ref_desc is not None else desc
                    else:
                        return ref_text
            return desc
    return None

async def check_blacklist_drugs(drugs, drug_dict, automaton):
    """
    Проверка препаратов в расстрельном списке с использованием drug_dict и Ахо–Корасик автомата.
    Для каждого препарата определяется, содержится ли он в списке, и извлекается описание.
    """
    results = []
    for drug in drugs:
        in_list = is_in_list_aho(drug, automaton)
        description = get_description_aho(drug, drug_dict, automaton) if in_list else None
        results.append({
            "preparation": drug,
            "in_blacklist": in_list,
            "blacklist_description": description
        })
    return results
