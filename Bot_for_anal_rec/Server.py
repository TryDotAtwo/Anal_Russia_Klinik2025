# app.py

from quart import Quart, request, jsonify
import asyncio
from Anal_GPT import initialize_globals, analyze_preparations

# Создание приложения Quart
app = Quart(__name__)

# Конфигурация
config = {
    "metadata_path": 'C:/Users/Иван Литвак/source/repos/Anal_Russia_Klinik2025/Anal_Russia_Klinik2025/MetaData.json',
    "pdf_folder": 'C:/Users/Иван Литвак/source/repos/Anal_Russia_Klinik2025/Anal_Russia_Klinik2025/Клинические_Рекомендации',
    "blacklist_json_path": "blacklist_drugs.json",
    "clinical_recommendations_json": "clinical_recommendations.json",
    "max_concurrent_pdf": 36,
    "max_pdf_workers": 36,
    "pdf_batch_size": 10,
    "json_indent": 2,
    "max_drug_workers": 36,
    "context_before": 2000,
    "context_after": 2000
}

@app.route('/analyze', methods=['POST'])
async def analyze_drugs():
    try:
        data = await request.get_json()
        preparations = data.get('preparations', [])
        if not preparations:
            return jsonify({"error": "Список препаратов пуст"}), 400

        print(f"Получен запрос на анализ препаратов: {preparations}")

        # Вызываем синхронную функцию анализа в отдельном потоке
        results = await asyncio.to_thread(analyze_preparations, preparations, config)

        print(f"Анализ завершён для {len(results)} препаратов")
        return jsonify(results), 200
    except Exception as e:
        print(f"Ошибка при обработке запроса: {str(e)}")
        return jsonify({"error": "Внутренняя ошибка сервера", "details": str(e)}), 500

async def run_server():
    """Асинхронный запуск сервера с инициализацией данных"""
    await initialize_globals(config)  # Инициализация общих данных один раз при запуске
    await app.run_task(host="0.0.0.0", port=5000, debug=True)

if __name__ == "__main__":
    asyncio.run(run_server())