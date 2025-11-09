from flask import Flask, request, jsonify
import requests
import torch
import librosa
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import tempfile
import os
from datetime import timedelta
import io
import soundfile as sf

app = Flask(__name__)

# Конфигурация jonatasgrosman/wav2vec2-large-xlsr-53-russian - основной, но можно просто попбробовать bond005/wav2vec2-large-ru-golos и почти любые wav2vec2 с https://huggingface.co/
# для других языков просото другая модель, скрипт вроде униаерсален
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"
CALLBACK_URL = "http://localhost:9090/api/v1/scripts/from-translate"  # публичный ip и не нужен

# Загрузка модели и процессора при старте
print("Загружаем модель и процессор...")
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

# ВЫБОР НА ЧЁМ РАБОТАТЬ cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, ort, xla, lazy, vulkan, mps, meta, hpu, mtia
model.eval()
model.to("cpu")

def format_timestamp(seconds):
    """Конвертирует секунды в формат ЧЧ:ММ:СС"""
    return str(timedelta(seconds=round(seconds)))

def process_audio_from_bytes(audio_bytes, original_filename):
    """Обрабатывает аудио из байтов и возвращает транскрипцию с временными метками"""
    
    # Создаем временный файл с правильным расширением
    file_ext = os.path.splitext(original_filename)[1].lower()
    if not file_ext:
        file_ext = '.wav'
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_path = tmp_file.name
    
    try:
        #частота 16kHz
        speech_array, sampling_rate = librosa.load(tmp_path, sr=16_000)
        duration = len(speech_array) / sampling_rate
        
        # Если аудио короткое, обрабатываем целиком можно удалить чтобы гарантировать одинаковый размер чанков
        if duration <= 30:
            inputs = processor(speech_array, sampling_rate=16_000, return_tensors="pt", padding=True)
            
            with torch.no_grad():
                logits = model(inputs.input_values).logits
                
            predicted_ids = torch.argmax(logits, dim=-1)
            predicted_sentence = processor.batch_decode(predicted_ids)[0]
            
            return [f"00:00:00-{format_timestamp(duration)}: {predicted_sentence}"]
        
        # Для длинных аудио разбиваем на чанки. 
        scripts = []
        chunk_size = 10 * sampling_rate  
        
        for i in range(0, len(speech_array), chunk_size):
            chunk = speech_array[i:i + chunk_size]
            chunk_duration_actual = len(chunk) / sampling_rate
            
            if chunk_duration_actual < 0.5:  # Пропускаем слишком короткие чанки, обычной это кряхтение всякое
                continue
                
            start_time = i / sampling_rate
            end_time = (i + len(chunk)) / sampling_rate
            
            inputs = processor(chunk, sampling_rate=16_000, return_tensors="pt", padding=True)
            
            with torch.no_grad():
                logits = model(inputs.input_values).logits
                
            predicted_ids = torch.argmax(logits, dim=-1)
            predicted_sentence = processor.batch_decode(predicted_ids)[0].strip()
            
            if predicted_sentence:  # Добавляем только непустые результаты
                scripts.append(f"{format_timestamp(start_time)}-{format_timestamp(end_time)}: {predicted_sentence}")
        
        return scripts
    
    finally:
        try:
            # Так нада
            os.unlink(tmp_path)
        except:
            pass

@app.route('/api/v1/scripts/to-translate/<id>', methods=['POST'])
def transcribe_audio(id):
    if 'file' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    file = request.files['file']
    
    # Желательно работать только с wav, с остальными не пробовал)))))))))))))
    if not file.filename.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.ogg')):
        return jsonify({"error": "Unsupported file format"}), 400
    
    try:
        # Читаем файл в память
        audio_bytes = file.read()
        
        if len(audio_bytes) == 0:
            return jsonify({"error": "Empty file"}), 400
            
        # Выполняем транскрипцию
        scripts = process_audio_from_bytes(audio_bytes, file.filename)
        
    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500


    callback_data = {
        "scripts": scripts,
        "count": len(scripts)
    }
    
    try:
        response = requests.post(CALLBACK_URL, json=callback_data, timeout=30)
        print(f"Callback response: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Callback failed: {str(e)}")
        # Не возвращаем ошибку, если callback не сработал, все равно показываем результат
        # return jsonify({"error": f"Callback failed: {str(e)}"}), 500

    return jsonify({
        "status": "success", 
        "id": id,
        "scripts": scripts,
        "count": len(scripts)
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "model": MODEL_ID})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9090, debug=False)