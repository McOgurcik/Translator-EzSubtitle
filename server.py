from flask import Flask, request, jsonify
import requests
import torch
import librosa
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import os
from datetime import timedelta
import glob
import threading
import tempfile
import soundfile as sf

app = Flask(__name__)

# Конфигурация
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"
CALLBACK_URL = "http://localhost:8080/api/v1/scripts/from-translate"  # Клиент на порту 8080

# Загрузка модели и процессора при старте
print("Загружаем модель и процессор...")
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
model.eval()
model.to("cpu")

def format_timestamp(seconds):
    """Конвертирует секунды в формат ЧЧ:ММ:СС"""
    return str(timedelta(seconds=round(seconds)))

def split_long_audio(audio_path, max_duration=10):
    """Разбивает длинное аудио на сегменты по max_duration секунд"""
    try:
        # Загружаем аудио
        audio_data, sample_rate = librosa.load(audio_path, sr=16_000)
        duration = len(audio_data) / sample_rate
        
        print(f"  Проверка длительности файла: {duration:.2f} секунд")
        
        # Если аудио короче max_duration, возвращаем как есть
        if duration <= max_duration:
            print(f"  Файл достаточно короткий, обрабатываем целиком")
            return [audio_data], [duration], sample_rate
        
        # Рассчитываем размер чанка в сэмплах
        chunk_size = int(max_duration * sample_rate)
        
        # Разбиваем на сегменты
        segments = []
        segment_durations = []
        
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            chunk_duration = len(chunk) / sample_rate
            
            # Пропускаем слишком короткие сегменты
            if chunk_duration < 0.5:
                continue
                
            segments.append(chunk)
            segment_durations.append(chunk_duration)
        
        print(f"  Разбит на {len(segments)} сегментов по ~{max_duration} секунд")
        return segments, segment_durations, sample_rate
        
    except Exception as e:
        print(f"  Ошибка при разбиении аудио {audio_path}: {str(e)}")
        return [], [], 0

def transcribe_audio_segment(audio_segment, sampling_rate):
    """Транскрибирует один сегмент аудио и возвращает текст"""
    try:
        inputs = processor(audio_segment, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            logits = model(inputs.input_values).logits
            
        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_sentence = processor.batch_decode(predicted_ids)[0].strip()
        
        return predicted_sentence
    except Exception as e:
        print(f"  Ошибка транскрипции сегмента: {str(e)}")
        return ""

def process_audio_file(file_path):
    """Обрабатывает аудиофайл, разбивая длинные файлы на сегменты"""
    try:
        # Сначала проверяем длительность файла
        audio_data, sampling_rate = librosa.load(file_path, sr=16_000)
        duration = len(audio_data) / sampling_rate
        
        print(f"Обработка файла: {os.path.basename(file_path)}")
        print(f"  Длительность: {duration:.2f} секунд")
        
        # Если файл длиннее 20 секунд, разбиваем на сегменты по 10 секунд
        if duration > 20:
            print(f"  Файл слишком длинный, разбиваем на сегменты по 10 секунд...")
            segments, durations, sr = split_long_audio(file_path, 10)
            
            results = []
            for i, (segment, seg_duration) in enumerate(zip(segments, durations)):
                print(f"  Транскрипция сегмента {i+1}/{len(segments)}...")
                text = transcribe_audio_segment(segment, sr)
                print(f"  Результат: {text[:50]}{'...' if len(text) > 50 else ''}")
                results.append((text, seg_duration))
            
            return results
        else:
            # Обрабатываем файл целиком
            print(f"  Транскрипция файла...")
            text = transcribe_audio_segment(audio_data, sampling_rate)
            print(f"  Результат: {text[:50]}{'...' if len(text) > 50 else ''}")
            return [(text, duration)]
            
    except Exception as e:
        print(f"  Ошибка обработки файла {file_path}: {str(e)}")
        return [("", 0)]

def get_sorted_segment_files(directory_path):
    """Получает отсортированный список сегментных файлов из директории"""
    pattern = os.path.join(directory_path, "segment_*.wav")
    segment_files = glob.glob(pattern)
    segment_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    return segment_files

def process_audio_async(id, directory_path):
    """Асинхронная обработка аудио"""
    try:
        print(f"Начата обработка для ID: {id}")
        print(f"Директория с сегментами: {directory_path}")
        
        segment_files = get_sorted_segment_files(directory_path)
        
        if not segment_files:
            print(f"Ошибка: не найдены сегментные файлы в {directory_path}")
            return
        
        print(f"Найдено {len(segment_files)} сегментных файлов")
        
        scripts = []
        total_duration = 0.0

        for i, segment_file in enumerate(segment_files):
            print(f"[{i+1}/{len(segment_files)}] Обработка: {os.path.basename(segment_file)}")
            
            # Обрабатываем файл (может вернуть несколько сегментов)
            segment_results = process_audio_file(segment_file)
            
            for text, duration in segment_results:
                if text:  # Добавляем только непустые результаты
                    start_time = total_duration
                    end_time = total_duration + duration
                    time_range = f"{format_timestamp(start_time)}-{format_timestamp(end_time)}"
                    scripts.append(f"{time_range}: {text}")
                    total_duration = end_time
                    print(f"  Добавлен сегмент: {time_range}")

        print(f"Обработка завершена. Всего сегментов: {len(scripts)}")
        print(f"Общая длительность: {total_duration:.2f} секунд")

        # Отправляем результат на callback URL
        callback_data = {
            "id": id,
            "scripts": scripts,
            "count": len(scripts)
        }
        
        callback_url = f"{CALLBACK_URL}/{id}"
        print(f"Отправка callback на: {callback_url}")
        try:
            response = requests.post(callback_url, json=callback_data, timeout=30)
            print(f"Callback отправлен, статус: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Ошибка отправки callback: {str(e)}")
            
    except Exception as e:
        print(f"Ошибка асинхронной обработки: {str(e)}")

@app.route('/api/v1/scripts/to-translate/<id>', methods=['POST'])
def transcribe_audio(id):
    data = request.get_json()
    if not data or 'path' not in data:
        return jsonify({"error": "No path provided"}), 400
    
    directory_path = data['path']
    
    if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
        return jsonify({"error": f"Directory not found: {directory_path}"}), 400
    
    print(f"Получен запрос на транскрипцию, ID: {id}")
    print(f"Путь к сегментам: {directory_path}")
    
    # Запускаем асинхронную обработку в отдельном потоке
    thread = threading.Thread(target=process_audio_async, args=(id, directory_path))
    thread.daemon = True
    thread.start()
    
    print(f"Запущена асинхронная обработка для ID: {id}")
    return '', 200

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "model": MODEL_ID})

if __name__ == '__main__':
    print("Сервер транскрипции запускается на порту 9090")
    print(f"Callback URL: {CALLBACK_URL}")
    app.run(host='0.0.0.0', port=9090, debug=False)