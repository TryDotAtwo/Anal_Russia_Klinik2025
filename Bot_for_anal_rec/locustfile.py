from locust import HttpUser, task, between
import json

class ClinicalAnalyzerUser(HttpUser):
    host = "http://localhost:5000"
    wait_time = between(1, 5)

    @task
    def analyze_drugs(self):
        response = self.client.post("/analyze", json={"preparations": [
            "Кагоцел", "Эргоферон", "Эргоферон", "Кагоцел", "Эргоферон", "Эргоферон", "Кагоцел", "Эргоферон", "Эргоферон", "Кагоцел", "Эргоферон", "Эргоферон", "Кагоцел", "Эргоферон", "Эргоферон", "Кагоцел", "Эргоферон", "Эргоферон"
        ]}, timeout=600)
        
        if response.status_code == 200:
            try:
                results = response.json()
                self.process_response(results)
            except json.JSONDecodeError:
                print("Ошибка декодирования ответа JSON")
        else:
            print(f"Ошибка запроса: {response.status_code}")

    def process_response(self, results):
        for result in results:
            preparation = result.get("preparation", "Неизвестно")
            blacklist_desc = result.get("blacklist_description", "Нет описания")
            comment = result.get("comment", "Нет комментария")
            
            total_mentions = result.get("analysis_summary", {}).get("total_mentions", 0)
            response_time = self.environment.stats.total.get_response_time_percentile(50)
            
            print(f"\nПрепарат: {preparation}")
            print(f"Пользователь: {self.environment.runner.user_count}")
            print(f"Время ответа: {response_time:.2f} мс")
            print(f"Найдено совпадений: {total_mentions}")
            print(f"Комментарий модели: {comment[:6]}...")
            print(f"Описание из черного списка: {blacklist_desc[:6]}...\n")

if __name__ == "__main__":
    import os
    os.system("locust -f locustfile.py")
