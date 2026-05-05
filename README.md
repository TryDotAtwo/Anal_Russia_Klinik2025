# Anal_Russia_Klinik2025  
[![Статистика клинорекомендаций](https://img.shields.io/badge/Клинорекомендаций-560-success?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiNmZmYiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48cGF0aCBkPSJNMTIgMkwzIDEySDIxTDEyIDIyWiIvPjwvc3ZnPg==)](https://github.com/TryDotAtwo/Anal_Russia_Klinik2025)


**Парсер и анализатор российских клинических рекомендаций и протоколов ведения больных (2024–2026)**  
Большая работа по сбору, очистке и классификации упоминаний лекарственных средств в официальных российских клинических рекомендациях.

Клинические рекомендации можно найти на сайте https://cr.minzdrav.gov.ru/clin-rec. 
В качестве основного источника информации использован расстрельный список препаратов https://encyclopatia.ru/wiki/Расстрельный_список_препаратов и сайт MedIq https://mediqlab.com/.


## Поиск осуществляется через совпадения слов или словоформ с последующей оценкой LLM контекста упоминания. 
Результаты можно найти по ссылке: https://docs.google.com/spreadsheets/d/1ez0kcEPb7fG8E_0f7F5LrCfPg0fsiuQpH1AMKid0gX8/edit?gid=671773976#gid=671773976
Прошлые результаты 2025: https://docs.google.com/spreadsheets/d/1H4c2ApuLdaliCj3b2HDTVYmmjIFCTOuWzcS6LjxclGM/edit?usp=sharing
Прошлые результаты 2024: https://docs.google.com/spreadsheets/d/1C0-pqnQktmtxNsOc3cU8fobhCf_nr1RKxfv1dNGzDTQ/edit?gid=1018405073#gid=1018405073

### Общая статистика (на 03 декабря 2025)

| Показатель                                  | Значение |
| ------------------------------------------- | -------- |
| Всего клинических рекомендаций              | **560**  |
| Всего LLM обнаружила сомнительного          |**3 778** |


### Валидные упоминания по источникам

| Источник  | Количество |
| --------- | ---------- |
| MedIQ     | **4307**   |
| Blacklist | **1431**   |
| Marker    | **716**    |
| **Итого** | **6454***  |

### Распределение по типам

| Тип                | Количество |
| ------------------ | ---------- |
| Recommendation     | **3088**   |
| Contraindication   | **116**    |
| Literature mention | **686**    |
| Error              | **1794**   |
| Unclear            | **4**      |
| **Итого**          | **5688**   |

### Топ уровней доказательности и силы рекомендаций

| Уровень         | Количество  |
| --------------- | ----------- |
| C5              | **1066**    |
| B2              | **262**     |
| C4              | **246**     |
| A1              | **200**     |
| B1              | **145**     |
| B3              | **140**     |
| A2              | **111**     |
| C3              | **82**      |
| A3              | **77**      |
| C2              | 58          |
| C1              | 26          |
| B4              | 27          |
| B5              | 14          |
| A               | 11          |
| B               | 11          |
| C               | 39          |
| A5              | 3           |
| A4              | 1           |
| "1","2","3","5" | 14 суммарно |




Если вы врач, фармаколог, исследователь или просто интересуетесь доказательной медициной в России — welcome to contribute! ⭐
