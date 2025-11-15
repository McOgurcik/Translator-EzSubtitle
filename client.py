from flask import Flask, request, jsonify
import requests
import json
import sys
import os
import threading
import time

# Клиентский Flask сервер для приема callback
client_app = Flask(__name__)
received_result = None
result_received_event = threading.Event()

@client_app.route('/api/v1/scripts/from-translate/<id>', methods=['POST'])
def handle_callback(id):
    global received_result
    received_result = request.get_json()
    print(f"Получен callback для ID: {id}")
    result_received_event.set()
    return '', 200

def run_client_server():
    print("Клиентский сервер запускается на порту 8080")
    client_app.run(host='0.0.0.0', port=8080, debug=False, use_reloader=False)

def send_audio_for_transcription(audio_path, server_url="http://localhost:9090", audio_id="test_001"):
    """Отправляет путь к аудиофайлу на сервер транскрипции"""
    url = f"{server_url}/api/v1/scripts/to-translate/{audio_id}"
    
    data = {
        "path": audio_path
    }
    
    print(f"Отправка запроса на сервер: {url}")
    print(f"ID: {audio_id}")
    print(f"Аудиофайл: {audio_path}")
    
    response = requests.post(url, json=data, timeout=10)
    
    if response.status_code == 200:
        print("Запрос принят сервером, начата обработка")
        return True
    else:
        print(f"Ошибка при отправке запроса: {response.status_code}")
        try:
            error_detail = response.json()
            print(f"Детали ошибки: {error_detail}")
        except:
            print(f"Сообщение: {response.text}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Использование: python client.py <аудиофайл> [server_url] [id_аудио]")
        print("Пример: python client.py audio.wav http://localhost:9090 test_001")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    # Параметры по умолчанию
    server_url = "http://localhost:9090"
    audio_id = "test_001"
    
    # Парсим аргументы
    for arg in sys.argv[2:]:
        if arg.startswith('http://') or arg.startswith('https://'):
            server_url = arg
        elif '=' in arg and arg.split('=')[0] == 'id':
            audio_id = arg.split('=')[1]
    
    if not os.path.exists(audio_file):
        print(f"Ошибка: файл {audio_file} не найден")
        sys.exit(1)
    
    # Проверяем расширение файла
    if not audio_file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.ogg')):
        print(f"Ошибка: неподдерживаемый формат файла: {audio_file}")
        sys.exit(1)
    
    # Запускаем клиентский сервер в отдельном потоке
    print("Запуск клиентского сервера для приема callback...")
    server_thread = threading.Thread(target=run_client_server)
    server_thread.daemon = True
    server_thread.start()
    
    # Даем серверу время запуститься
    time.sleep(2)
    
    # Отправляем запрос на сервер
    success = send_audio_for_transcription(audio_file, server_url, audio_id)
    
    if not success:
        print("Ошибка: не удалось отправить запрос на сервер")
        sys.exit(1)
    
    print("Ожидание результата транскрипции...")
    
    # Ждем результат с таймаутом 10 минут
    timeout = 600
    if result_received_event.wait(timeout):
        # Выводим полученный результат
        print("\n" + "="*50)
        print("РЕЗУЛЬТАТ ТРАНСКРИПЦИИ:")
        print("="*50)
        print(json.dumps(received_result, ensure_ascii=False, indent=2))
    else:
        print("Ошибка: таймаут ожидания результата транскрипции")

if __name__ == '__main__':
    main()