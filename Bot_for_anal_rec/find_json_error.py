import os
os.chdir(r'C:\Users\Иван Литвак\source\repos\Bot_for_anal_rec\Bot_for_anal_rec')
with open('reports/llm/llm_gold_40.json', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    # Показать строки вокруг 1116 (0-indexed так 1115)
    for i in range(max(0, 1113), min(len(lines), 1118)):
        print(f"Line {i+1}:")
        print(lines[i])
