from flask import Flask, request, jsonify
import requests
import json
import sys
import os
import tempfile
import librosa
import soundfile as sf
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
    result_received_event.set()
    return '', 200

def run_client_server():
    client_app.run(host='0.0.0.0', port=8080, debug=False, use_reloader=False)

def split_audio_file(audio_path, chunk_duration=10, output_dir=None):
    """Разбивает аудиофайл на чанки заданной длительности"""
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="audio_segments_")
    
    audio_data, sample_rate = librosa.load(audio_path, sr=None)
    chunk_size = int(chunk_duration * sample_rate)
    segment_count = 0
    
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i:i + chunk_size]
        chunk_duration_actual = len(chunk) / sample_rate
        
        if chunk_duration_actual < 0.5:
            continue
            
        segment_filename = f"segment_{segment_count}.wav"
        segment_path = os.path.join(output_dir, segment_filename)
        sf.write(segment_path, chunk, sample_rate)
        segment_count += 1
    
    return output_dir

def send_audio_path_for_transcription(segments_dir, server_url="http://localhost:9090", audio_id="test_001"):
    """Отправляет путь к директории с сегментами на сервер транскрипции"""
    url = f"{server_url}/api/v1/scripts/to-translate/{audio_id}"
    
    data = {
        "path": segments_dir
    }
    
    response = requests.post(url, json=data, timeout=10)
    return response.status_code == 200

def cleanup_directory(directory_path):
    """Очищает временную директорию"""
    try:
        import shutil
        shutil.rmtree(directory_path)
    except:
        pass

def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <audio_file> [chunk_duration] [server_url] [audio_id]")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    chunk_duration = 10
    server_url = "http://localhost:9090"
    audio_id = "test_001"
    
    for arg in sys.argv[2:]:
        if arg.isdigit():
            chunk_duration = int(arg)
        elif arg.startswith('http://') or arg.startswith('https://'):
            server_url = arg
        elif '=' in arg and arg.split('=')[0] == 'id':
            audio_id = arg.split('=')[1]
    
    if not os.path.exists(audio_file):
        print(f"Error: File {audio_file} not found")
        sys.exit(1)
    
    # Запускаем клиентский сервер в отдельном потоке
    server_thread = threading.Thread(target=run_client_server)
    server_thread.daemon = True
    server_thread.start()
    
    # Даем серверу время запуститься
    time.sleep(2)
    
    # Разбиваем аудио на сегменты
    segments_dir = split_audio_file(audio_file, chunk_duration)
    
    if not segments_dir or len(os.listdir(segments_dir)) == 0:
        print("Error: Failed to split audio file into segments")
        sys.exit(1)
    
    # Отправляем запрос на сервер
    success = send_audio_path_for_transcription(segments_dir, server_url, audio_id)
    
    if not success:
        print("Error: Failed to send request to server")
        cleanup_directory(segments_dir)
        sys.exit(1)
    
    print("Request accepted by server. Waiting for transcription result...")
    
    # Ждем результат с таймаутом 10 минут
    timeout = 600
    if result_received_event.wait(timeout):
        print(json.dumps(received_result, ensure_ascii=False, indent=2))
    else:
        print("Error: Timeout waiting for transcription result")
    
    cleanup_directory(segments_dir)

if __name__ == '__main__':
    main()