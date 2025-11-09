import requests
import json
import sys
import os
import time

def send_audio_for_transcription(audio_file_path, server_url="http://localhost:9090", audio_id="test_001"):
    """
    Отправляет аудиофайл на сервер транскрипции и возвращает результат
    """
    
    if not os.path.exists(audio_file_path):
        print(f"Ошибка: файл {audio_file_path} не найден")
        return None
    
    url = f"{server_url}/api/v1/scripts/to-translate/{audio_id}"
    
    try:
        with open(audio_file_path, 'rb') as audio_file:
            files = {'file': (os.path.basename(audio_file_path), audio_file, 'audio/wav')}
            
            print(f"Отправка файла {audio_file_path} на сервер...")
            start_time = time.time()
            response = requests.post(url, files=files, timeout=600)
            processing_time = time.time() - start_time
            
        if response.status_code == 200:
            result = response.json()
            print(f"Транскрипция успешно выполнена за {processing_time:.2f} секунд!")
            print(f"ID: {result.get('id')}")
            print(f"Количество фраз: {result.get('count')}")
            print("\nРезультаты транскрипции:")
            
            for i, script in enumerate(result.get('scripts', [])):
                print(f"{i+1}. {script}")
            
            return result
        else:
            print(f"Ошибка сервера: {response.status_code}")
            try:
                error_detail = response.json()
                print(f"Детали ошибки: {error_detail}")
            except:
                print(f"Сообщение: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Ошибка подключения: {str(e)}")
        return None
    except Exception as e:
        print(f"Неожиданная ошибка: {str(e)}")
        return None

def health_check(server_url="http://localhost:9090"):
    """Проверяет статус сервера"""
    try:
        response = requests.get(f"{server_url}/health", timeout=5)
        if response.status_code == 200:
            status = response.json()
            print(f"Сервер работает нормально")
            print(f"Модель: {status.get('model')}")
            return True
        else:
            print(f"Сервер вернул ошибку: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Не удалось подключиться к серверу: {str(e)}")
        return False

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Использование: python client.py <путь_к_аудиофайлу> [server_url] [audio_id]")
        print("Пример: python client.py audio.wav http://localhost:9090 test_001")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    server_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:9090"
    audio_id = sys.argv[3] if len(sys.argv) > 3 else "test_001"
    
    # Проверяем здоровье сервера
    if not health_check(server_url):
        print("Сервер недоступен. Запустите server.py сначала.")
        sys.exit(1)
    
    # Отправляем аудио на транскрипцию
    result = send_audio_for_transcription(audio_file, server_url, audio_id)
    
    if result is None:
        sys.exit(1)