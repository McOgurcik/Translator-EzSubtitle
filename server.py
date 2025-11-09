from flask import Flask, request, jsonify
import requests
import torch
import librosa
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import os
from datetime import timedelta, datetime
import glob
import threading
import logging
from logging.handlers import RotatingFileHandler
import json

app = Flask(__name__)

# Конфигурация
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"
CALLBACK_URL = "http://localhost:8080/api/v1/scripts/from-translate"

# Настройка логирования
def setup_logging():
    """Настройка системы логирования"""
    # Создаем директорию для логов если ее нет
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Формат логов
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s'
    )
    
    # Основной логгер приложения
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Файловый обработчик с ротацией
    file_handler = RotatingFileHandler(
        f'{log_dir}/app.log',
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Консольный обработчик
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Инициализация логирования
setup_logging()
logger = logging.getLogger(__name__)

# Загрузка модели и процессора при старте
logger.info("Загружаем модель и процессор...")
try:
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    model.eval()
    model.to("cpu")
    logger.info(f"Модель {MODEL_ID} успешно загружена на CPU")
except Exception as e:
    logger.error(f"Ошибка загрузки модели: {str(e)}")
    raise

def format_timestamp(seconds):
    """Конвертирует секунды в формат ЧЧ:ММ:СС"""
    return str(timedelta(seconds=round(seconds)))

def log_processing_start(id, directory_path):
    """Логирование начала обработки"""
    logger.info(
        f"Начало обработки - ID: {id}, Директория: {directory_path}, "
        f"Время: {datetime.now().isoformat()}"
    )

def log_processing_end(id, scripts_count, total_duration):
    """Логирование завершения обработки"""
    logger.info(
        f"Завершение обработки - ID: {id}, "
        f"Фраз: {scripts_count}, Длительность: {total_duration:.2f}с"
    )

def log_segment_processing(segment_file, duration, text):
    """Логирование обработки сегмента"""
    if text:
        logger.info(
            f"Сегмент обработан - Файл: {os.path.basename(segment_file)}, "
            f"Длительность: {duration:.2f}с, Текст: '{text}'"
        )
    else:
        logger.warning(
            f"Пустой результат - Файл: {os.path.basename(segment_file)}, "
            f"Длительность: {duration:.2f}с"
        )

def transcribe_audio_file(file_path):
    """Транскрибирует один аудиофайл и возвращает текст и длительность"""
    try:
        start_time = datetime.now()
        speech_array, sampling_rate = librosa.load(file_path, sr=16_000)
        duration = len(speech_array) / sampling_rate
        
        inputs = processor(speech_array, sampling_rate=16_000, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            logits = model(inputs.input_values).logits
            
        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_sentence = processor.batch_decode(predicted_ids)[0].strip()
        
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.debug(
            f"Файл обработан - {os.path.basename(file_path)}: "
            f"{processing_time:.2f}с, {duration:.2f}с аудио"
        )
        
        return predicted_sentence, duration
    except Exception as e:
        logger.error(
            f"Ошибка обработки файла {file_path}: {str(e)}",
            exc_info=True
        )
        return "", 0

def get_sorted_segment_files(directory_path):
    """Получает отсортированный список сегментных файлов из директории"""
    try:
        pattern = os.path.join(directory_path, "segment_*.wav")
        segment_files = glob.glob(pattern)
        segment_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        
        logger.info(
            f"Найдено сегментов: {len(segment_files)} в {directory_path}"
        )
        return segment_files
    except Exception as e:
        logger.error(
            f"Ошибка поиска сегментов в {directory_path}: {str(e)}",
            exc_info=True
        )
        return []

def process_audio_async(id, directory_path):
    """Асинхронная обработка аудио"""
    logger.info(f"Запуск асинхронной обработки для ID: {id}")
    
    try:
        log_processing_start(id, directory_path)
        segment_files = get_sorted_segment_files(directory_path)
        
        if not segment_files:
            logger.warning(f"Сегменты не найдены в {directory_path}")
            return
        
        scripts = []
        total_duration = 0.0
        processed_segments = 0

        for segment_file in segment_files:
            text, duration = transcribe_audio_file(segment_file)
            processed_segments += 1
            
            log_segment_processing(segment_file, duration, text)
            
            if text:
                start_time = total_duration
                end_time = total_duration + duration
                time_range = f"{format_timestamp(start_time)}-{format_timestamp(end_time)}"
                scripts.append(f"{time_range}: {text}")
                total_duration = end_time

        log_processing_end(id, len(scripts), total_duration)
        
        # Отправляем результат на callback URL
        callback_data = {
            "id": id,
            "scripts": scripts,
            "count": len(scripts)
        }
        
        callback_url = f"{CALLBACK_URL}/{id}"
        try:
            logger.info(f"Отправка callback для ID: {id} на {callback_url}")
            logger.debug(f"Callback данные: {json.dumps(callback_data, ensure_ascii=False)}")
            
            response = requests.post(callback_url, json=callback_data, timeout=30)
            response.raise_for_status()
            
            logger.info(
                f"Callback успешен - ID: {id}, "
                f"Статус: {response.status_code}, Фраз: {len(scripts)}"
            )
            
        except requests.exceptions.RequestException as e:
            logger.error(
                f"Callback ошибка - ID: {id}: {str(e)}",
                exc_info=True
            )
        except Exception as e:
            logger.error(
                f"Неожиданная ошибка callback - ID: {id}: {str(e)}",
                exc_info=True
            )
            
    except Exception as e:
        logger.error(
            f"Ошибка асинхронной обработки - ID: {id}: {str(e)}",
            exc_info=True
        )

@app.before_request
def log_request_info():
    """Логирование входящих запросов"""
    logger.info(
        f"Входящий запрос - Метод: {request.method}, "
        f"Путь: {request.path}, IP: {request.remote_addr}"
    )

@app.after_request
def log_response_info(response):
    """Логирование исходящих ответов"""
    logger.info(
        f"Исходящий ответ - Метод: {request.method}, "
        f"Путь: {request.path}, Статус: {response.status_code}"
    )
    return response

@app.route('/api/v1/scripts/to-translate/<id>', methods=['POST'])
def transcribe_audio(id):
    """Основной эндпоинт для транскрипции аудио"""
    logger.info(f"Обработка запроса транскрипции - ID: {id}")
    
    try:
        data = request.get_json()
        if not data or 'path' not in data:
            logger.warning(f"Неверные данные запроса - ID: {id}")
            return jsonify({"error": "No path provided"}), 400
        
        directory_path = data['path']
        logger.info(f"Получен путь для обработки - ID: {id}, Путь: {directory_path}")
        
        if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
            logger.error(f"Директория не найдена - ID: {id}, Путь: {directory_path}")
            return jsonify({"error": f"Directory not found: {directory_path}"}), 400
        
        # Запускаем асинхронную обработку
        thread = threading.Thread(
            target=process_audio_async, 
            args=(id, directory_path),
            name=f"TranscribeThread-{id}"
        )
        thread.daemon = True
        thread.start()
        
        logger.info(f"Асинхронная обработка запущена - ID: {id}")
        return '', 200
        
    except Exception as e:
        logger.error(
            f"Ошибка в эндпоинте транскрипции - ID: {id}: {str(e)}",
            exc_info=True
        )
        return jsonify({"error": "Internal server error"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Эндпоинт для проверки здоровья сервиса"""
    logger.debug("Запрос health check")
    return jsonify({
        "status": "healthy", 
        "model": MODEL_ID,
        "timestamp": datetime.now().isoformat()
    })

@app.errorhandler(Exception)
def handle_exception(e):
    """Глобальный обработчик исключений"""
    logger.error(
        f"Необработанное исключение: {str(e)}",
        exc_info=True
    )
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    logger.info("Запуск сервера транскрипции на порту 9090")
    try:
        app.run(host='0.0.0.0', port=9090, debug=False)
    except Exception as e:
        logger.critical(f"Ошибка запуска сервера: {str(e)}", exc_info=True)
        raise