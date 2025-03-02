import asyncio
import os
import re
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import g4f
import fitz
from bs4 import BeautifulSoup
import g4f  # gpt4free
import re
import time
import requests
from bs4 import BeautifulSoup
import asyncio
import aiohttp
import json
from dotenv import load_dotenv
load_dotenv()

# Устанавливаем правильный event loop для Windows
if asyncio.get_event_loop_policy().__class__.__name__ != "WindowsSelectorEventLoopPolicy":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())



# --- Конфигурация ---
PDF_FOLDER = 'C:/Users/Иван Литвак/source/repos/Anal_Russia_Klinik2025/Anal_Russia_Klinik2025/Клинические_Рекомендации'
BLACKLIST_PATH = 'C:/Users/Иван Литвак/source/repos/Anal_Russia_Klinik2025/Anal_Russia_Klinik2025/Расстрельный список препаратов — Encyclopedia Pathologica.html'

# --- Глобальные переменные ---
clinical_data = {}  # {pdf_id: {"text": str, "link": str}}
blacklist_html = ""
g4f.debug.logging = False

# --- Инициализация данных ---
async def load_data():
    global clinical_data, blacklist_html
    
    # Загрузка клинических рекомендаций
    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith('.pdf'):
            pdf_id = os.path.splitext(filename)[0]
            doc = fitz.open(os.path.join(PDF_FOLDER, filename))
            text = ""
            for page in doc:
                text += page.get_text("text")
            clinical_data[pdf_id] = {
                "text": re.sub(r'\s+', ' ', text).strip(),
                "link": f"https://cr.minzdrav.gov.ru/view-cr/{pdf_id.split('_')[0]}"
            }
    
    # Загрузка расстрельного списка
    with open(BLACKLIST_PATH, 'r', encoding='utf-8') as f:
        blacklist_html = f.read()



# --- Обработчики Telegram ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 Введите названия препаратов через запятую:")

async def handle_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    drugs = [d.strip() for d in update.message.text.split(',')]
    
    for drug in drugs:
        await update.message.reply_text(f"🔬 Анализирую {drug}...")
        
        # Параллельные запросы
        clinical_task = asyncio.create_task(analyze_clinical_recommendations(drug))
        blacklist_task = asyncio.create_task(check_blacklist(drug))
        
        clinical_results = await clinical_task
        blacklist_result = await blacklist_task
        
        # Формирование ответа
        response = f"💊 *{drug.upper()}*\n\n"
        
        if clinical_results:
            response += "📚 *Клинические рекомендации:*\n"
            for res in clinical_results:
                response += f"- [{res['name']}]({res['link']})\n{res['analysis']}\n\n"
        else:
            response += "❌ Не найден в клинических рекомендациях\n\n"
        
        if blacklist_result:
            response += f"⚠️ *Расстрельный список:*\n{blacklist_result[:500]}..."
        else:
            response += "✅ Отсутствует в расстрельном списке"
        
        await update.message.reply_markdown(response)

# --- Запуск приложения ---
def main():
    application = Application.builder().token("TELEGRAM_BOT_TOKEN").build()
    
    # Инициализация данных
    loop = asyncio.get_event_loop()
    loop.run_until_complete(load_data())
    
    # Регистрация обработчиков
    application.add_handler(CommandHandler('start', start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_input))
    
    application.run_polling()

if __name__ == '__main__':
    main()