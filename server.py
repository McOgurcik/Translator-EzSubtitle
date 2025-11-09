from flask import Flask, request, jsonify
import requests
import torch
import librosa
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import tempfile
import os
from datetime import timedelta

app = Flask(__name__)

# Конфигурация
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"
CALLBACK_URL = "http://localhost:9090/api/v1/scripts/from-translate"


print("Загружаем модель и процессор...")
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

model.eval()
model.to("cpu")

def format_timestamp(seconds):
    """Конвертирует секунды в формат ЧЧ:ММ:СС"""
    return str(timedelta(seconds=round(seconds)))

def transcribe_audio_chunk(audio_bytes, original_filename):
    """Транскрибирует один чанк аудио и возвращает текст"""
    

    file_ext = os.path.splitext(original_filename)[1].lower()
    if not file_ext:
        file_ext = '.wav'
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_path = tmp_file.name
    
    try:
        # Загружаем аудио с частотой 16kHz
        speech_array, sampling_rate = librosa.load(tmp_path, sr=16_000)
        duration = len(speech_array) / sampling_rate
        
        # Обрабатываем весь аудиофайл как один чанк
        inputs = processor(speech_array, sampling_rate=16_000, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            logits = model(inputs.input_values).logits
            
        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_sentence = processor.batch_decode(predicted_ids)[0].strip()
        
        return predicted_sentence, duration
    
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass

@app.route('/api/v1/scripts/to-translate/<id>', methods=['POST'])
def transcribe_audio(id):
    # Получаем список файлов
    files = request.files.getlist('files')
    if not files:
        return jsonify({"error": "No audio files provided"}), 400

    scripts = []
    total_duration = 0.0

    # Обрабатываем каждый файл последовательно
    for i, file in enumerate(files):
        if not file.filename:
            continue
            
        if not file.filename.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.ogg')):
            return jsonify({"error": f"Unsupported file format: {file.filename}"}), 400
        
        try:

            audio_bytes = file.read()
            
            if len(audio_bytes) == 0:
                continue
                
            text, duration = transcribe_audio_chunk(audio_bytes, file.filename)
            

            start_time = total_duration
            end_time = total_duration + duration
            

            time_range = f"{format_timestamp(start_time)}-{format_timestamp(end_time)}"
            scripts.append(f"{time_range}: {text}")
            

            total_duration = end_time
            
            print(f"Обработан чанк {i+1}/{len(files)}: {time_range} - {text[:50]}...")
            
        except Exception as e:
            return jsonify({"error": f"Processing failed for file {file.filename}: {str(e)}"}), 500

    # Отправляем результат на callback-URL
    callback_data = {
        "scripts": scripts,
        "count": len(scripts)
    }
    
    try:
        response = requests.post(CALLBACK_URL, json=callback_data, timeout=30)
        print(f"Callback response: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Callback failed: {str(e)}")
        # Не возвращаем ошибку, если callback не сработал

    return jsonify({
        "status": "success", 
        "id": id,
        "scripts": scripts,
        "count": len(scripts),
        "total_duration": total_duration
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "model": MODEL_ID})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9090, debug=False)