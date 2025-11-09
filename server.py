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

app = Flask(__name__)

# Конфигурация
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"
CALLBACK_URL = "http://localhost:8080/api/v1/scripts/from-translate"  # Клиент слушает на 8080

# Загрузка модели и процессора при старте
print("Загружаем модель и процессор...")
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
model.eval()
model.to("cpu")

def format_timestamp(seconds):
    """Конвертирует секунды в формат ЧЧ:ММ:СС"""
    return str(timedelta(seconds=round(seconds)))

def transcribe_audio_file(file_path):
    """Транскрибирует один аудиофайл и возвращает текст и длительность"""
    try:
        speech_array, sampling_rate = librosa.load(file_path, sr=16_000)
        duration = len(speech_array) / sampling_rate
        
        inputs = processor(speech_array, sampling_rate=16_000, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            logits = model(inputs.input_values).logits
            
        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_sentence = processor.batch_decode(predicted_ids)[0].strip()
        
        return predicted_sentence, duration
    except Exception as e:
        print(f"Ошибка обработки файла {file_path}: {str(e)}")
        return "", 0

def get_sorted_segment_files(directory_path):
    """Получает отсортированный список сегментных файлов из директории"""
    pattern = os.path.join(directory_path, "segment_*.wav")
    segment_files = glob.glob(pattern)
    segment_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    return segment_files

def process_audio_async(id, directory_path):
    """Асинхронная обработка аудио"""
    try:
        segment_files = get_sorted_segment_files(directory_path)
        
        if not segment_files:
            print(f"No segment files found in {directory_path}")
            return
        
        scripts = []
        total_duration = 0.0

        for segment_file in segment_files:
            text, duration = transcribe_audio_file(segment_file)
            
            if text:  # Добавляем только непустые результаты
                start_time = total_duration
                end_time = total_duration + duration
                time_range = f"{format_timestamp(start_time)}-{format_timestamp(end_time)}"
                scripts.append(f"{time_range}: {text}")
                total_duration = end_time

        # Отправляем результат на callback URL
        callback_data = {
            "id": id,
            "scripts": scripts,
            "count": len(scripts)
        }
        
        callback_url = f"{CALLBACK_URL}/{id}"
        try:
            response = requests.post(callback_url, json=callback_data, timeout=30)
            print(f"Callback sent to {callback_url}, status: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Callback failed: {str(e)}")
            
    except Exception as e:
        print(f"Async processing failed: {str(e)}")

@app.route('/api/v1/scripts/to-translate/<id>', methods=['POST'])
def transcribe_audio(id):
    data = request.get_json()
    if not data or 'path' not in data:
        return jsonify({"error": "No path provided"}), 400
    
    directory_path = data['path']
    
    if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
        return jsonify({"error": f"Directory not found: {directory_path}"}), 400
    
    #асинхронно
    thread = threading.Thread(target=process_audio_async, args=(id, directory_path))
    thread.daemon = True
    thread.start()
    
    print(f"Started async processing for ID: {id}")
    return '', 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9090, debug=False)