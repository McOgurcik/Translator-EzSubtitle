from flask import Flask, request, jsonify
import requests
import torch
import librosa
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import os
from datetime import timedelta, datetime
import threading
import tempfile
import soundfile as sf
import noisereduce as nr
from scipy import signal
import logging
import sys
from logging.handlers import RotatingFileHandler

app = Flask(__name__)

# Конфигурация
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"
CALLBACK_URL = "http://localhost:8080/api/v1/scripts/from-translate"

# Настройка логирования
def setup_logging():
    """Настройка системы логирования"""
    # Создаем логгер
    logger = logging.getLogger('audio_transcriptor')
    logger.setLevel(logging.INFO)
    
    # Форматтер для логов
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Обработчик для консоли
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Обработчик для файла с ротацией
    file_handler = RotatingFileHandler(
        'audio_transcriptor.log',
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Добавляем обработчики к логгеру
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Инициализация логгера
logger = setup_logging()

# Загрузка модели и процессора при старте
logger.info("Загружаем модель и процессор...")
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
model.eval()
model.to("cpu")
logger.info("Модель и процессор успешно загружены")

def format_timestamp(seconds):
    """Конвертирует секунды в формат ЧЧ:ММ:СС"""
    return str(timedelta(seconds=round(seconds)))

def preprocess_audio(audio_data, sample_rate):
    """
    Предобработка аудио для улучшения качества транскрипции
    """
    logger.debug("Начата предобработка аудио")
    
    # 1. Конвертируем в моно, если стерео
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
        logger.debug("Конвертирован в моно")
    
    # 2. Ресемплинг до 16 кГц (требование модели)
    if sample_rate != 16000:
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000
        logger.debug("Ресемплинг до 16 кГц")
    
    # 3. Шумоподавление
    try:
        # Оцениваем шум по первой секунде (предполагаем, что это тишина/шум)
        noise_sample = audio_data[:min(16000, len(audio_data))]
        audio_data = nr.reduce_noise(
            y=audio_data, 
            sr=sample_rate,
            y_noise=noise_sample,
            prop_decrease=0.75,
            stationary=False
        )
        logger.debug("Применено шумоподавление")
    except Exception as e:
        logger.warning(f"Ошибка шумоподавления: {e}")
    
    # 4. Нормализация громкости
    rms = np.sqrt(np.mean(audio_data**2))
    if rms > 0:
        target_rms = 0.1  # Целевой уровень громкости
        audio_data = audio_data * (target_rms / rms)
        # Ограничиваем пики чтобы избежать клиппинга
        audio_data = np.clip(audio_data, -0.99, 0.99)
        logger.debug(f"Нормализация громкости (RMS: {rms:.4f} -> {np.sqrt(np.mean(audio_data**2)):.4f})")
    
    # 5. Фильтр высоких частот для удаления низкочастотного шума
    nyquist = sample_rate / 2
    highpass_cutoff = 80  # Hz
    if highpass_cutoff < nyquist:
        sos = signal.butter(4, highpass_cutoff/nyquist, btype='highpass', output='sos')
        audio_data = signal.sosfilt(sos, audio_data)
        logger.debug(f"Применен ВЧ-фильтр ({highpass_cutoff} Гц)")
    
    # 6. Компрессия динамического диапазона (мягкая)
    threshold = 0.5
    ratio = 2.0
    compressed = np.where(
        np.abs(audio_data) > threshold,
        np.sign(audio_data) * (threshold + (np.abs(audio_data) - threshold) / ratio),
        audio_data
    )
    # Плавно смешиваем с оригиналом чтобы не искажать слишком сильно
    mix = 0.3
    audio_data = audio_data * (1 - mix) + compressed * mix
    logger.debug("Применена мягкая компрессия")
    
    return audio_data, sample_rate

def adaptive_split_segment(audio_data, sample_rate, start_time, max_segment_duration=20, recursion_level=0):
    """
    Рекурсивно разделяет сегмент адаптивными методами
    """
    duration = len(audio_data) / sample_rate
    
    # Базовый случай: сегмент достаточно короткий
    if duration <= max_segment_duration:
        return [(audio_data, duration, start_time)]
    
    logger.info(f"Рекурсия {recursion_level}: сегмент {duration:.2f} сек слишком длинный, адаптивное разделение...")
    
    # Параметры для адаптивного разделения (более чувствительные на глубоких уровнях рекурсии)
    silence_params = [
        {"min_silence_duration": 0.8 - recursion_level * 0.2, "silence_threshold": -35 + recursion_level * 5, "padding": 0.2},
        {"min_silence_duration": 0.5 - recursion_level * 0.1, "silence_threshold": -30 + recursion_level * 5, "padding": 0.1},
        {"min_silence_duration": 0.3 - recursion_level * 0.05, "silence_threshold": -25 + recursion_level * 5, "padding": 0.05},
    ]
    
    # Ограничиваем минимальные значения
    for params in silence_params:
        params["min_silence_duration"] = max(0.1, params["min_silence_duration"])
        params["silence_threshold"] = min(-15, params["silence_threshold"])
    
    segments_found = []
    
    for i, params in enumerate(silence_params):
        logger.debug(f"Попытка {i+1} с параметрами: пауза={params['min_silence_duration']:.2f}с, порог={params['silence_threshold']}dB")
        
        segments = _split_audio_data_by_silence(audio_data, sample_rate, **params)
        
        # Проверяем, получилось ли разбить на несколько сегментов
        if len(segments) > 1:
            logger.info(f"Успешно разделено на {len(segments)} подсегментов")
            
            # Рекурсивно обрабатываем каждый подсегмент
            all_subsegments = []
            for segment, seg_duration, seg_start in segments:
                # Корректируем временную метку относительно исходного начала
                adjusted_start = start_time + seg_start
                subsegments = adaptive_split_segment(
                    segment, sample_rate, adjusted_start, 
                    max_segment_duration, recursion_level + 1
                )
                all_subsegments.extend(subsegments)
            
            return all_subsegments
    
    # Если адаптивное разделение не удалось, используем интеллектуальное разбиение по времени
    logger.info("Адаптивное разделение не удалось, используем интеллектуальное временное разделение")
    return intelligent_time_split(audio_data, sample_rate, start_time, max_segment_duration)

def intelligent_time_split(audio_data, sample_rate, start_time, max_segment_duration=20):
    """
    Интеллектуальное разделение по времени с поиском естественных границ
    """
    duration = len(audio_data) / sample_rate
    
    # Ищем точки с минимальной энергией для разбиения
    energy = librosa.feature.rms(y=audio_data, frame_length=1024, hop_length=256)[0]
    
    # Нормализуем энергию
    energy_normalized = (energy - np.min(energy)) / (np.max(energy) - np.min(energy))
    
    # Определяем количество сегментов
    num_segments = max(2, int(np.ceil(duration / max_segment_duration)))
    segment_duration = duration / num_segments
    
    segments = []
    
    for i in range(num_segments):
        segment_start_time = i * segment_duration
        segment_end_time = (i + 1) * segment_duration
        
        # Находим лучшую точку разбиения вокруг границы сегмента
        search_start = max(0, int((segment_start_time - 2) * sample_rate / 256))
        search_end = min(len(energy_normalized), int((segment_start_time + 2) * sample_rate / 256))
        
        if search_end > search_start:
            # Ищем локальный минимум энергии в окрестности границы
            search_window = energy_normalized[search_start:search_end]
            if len(search_window) > 0:
                min_index = np.argmin(search_window)
                best_split_time = (search_start + min_index) * 256 / sample_rate
            else:
                best_split_time = segment_start_time
        else:
            best_split_time = segment_start_time
        
        # Ограничиваем разбиение границами сегмента
        best_split_time = max(segment_start_time, min(segment_end_time, best_split_time))
        
        # Вычисляем сэмплы для сегмента
        start_sample = int(best_split_time * sample_rate)
        end_sample = int(segment_end_time * sample_rate) if i < num_segments - 1 else len(audio_data)
        
        segment = audio_data[start_sample:end_sample]
        segment_duration_actual = len(segment) / sample_rate
        
        if segment_duration_actual > 0.1:  # Минимальная длительность
            segments.append((segment, segment_duration_actual, start_time + best_split_time))
    
    logger.info(f"Интеллектуальное разделение на {len(segments)} сегментов")
    return segments

def _split_audio_data_by_silence(audio_data, sample_rate, min_silence_duration=0.5, silence_threshold=-40, padding=0.1):
    """
    Внутренняя функция для разделения аудиоданных по паузам с заданными параметрами
    """
    # Вычисляем энергию сигнала (громкость)
    energy = librosa.feature.rms(y=audio_data, frame_length=1024, hop_length=256)[0]
    
    # Конвертируем энергию в dB
    energy_db = librosa.amplitude_to_db(energy, ref=np.max)
    
    # Находим сегменты тишины
    silent_segments = []
    is_silent = False
    silent_start = 0
    
    for i, db in enumerate(energy_db):
        time_pos = i * 256 / sample_rate  # Текущее время в секундах
        
        if db < silence_threshold and not is_silent:
            # Начало тишины
            is_silent = True
            silent_start = time_pos
        elif db >= silence_threshold and is_silent:
            # Конец тишины
            silent_end = time_pos
            silent_duration = silent_end - silent_start
            
            if silent_duration >= min_silence_duration:
                silent_segments.append((silent_start, silent_end))
            
            is_silent = False
    
    # Обрабатываем случай, если аудио заканчивается тишиной
    if is_silent:
        silent_end = len(energy_db) * 256 / sample_rate
        silent_duration = silent_end - silent_start
        if silent_duration >= min_silence_duration:
            silent_segments.append((silent_start, silent_end))
    
    # Создаем сегменты речи между паузам
    speech_segments = []
    prev_end = 0
    
    for silent_start, silent_end in silent_segments:
        # Добавляем сегмент от предыдущего конца до начала текущей паузы
        segment_start = max(0, prev_end - padding)
        segment_end = silent_start + padding
        
        if segment_end > segment_start and (segment_end - segment_start) > 0.1:  # Минимальная длительность 0.1 сек
            start_sample = int(segment_start * sample_rate)
            end_sample = int(segment_end * sample_rate)
            segment = audio_data[start_sample:end_sample]
            segment_duration = len(segment) / sample_rate
            speech_segments.append((segment, segment_duration, segment_start))
        
        prev_end = silent_end
    
    # Добавляем последний сегмент (после последней паузы)
    if prev_end < len(audio_data) / sample_rate:
        segment_start = max(0, prev_end - padding)
        segment_end = len(audio_data) / sample_rate
        start_sample = int(segment_start * sample_rate)
        end_sample = int(segment_end * sample_rate)
        segment = audio_data[start_sample:end_sample]
        segment_duration = len(segment) / sample_rate
        speech_segments.append((segment, segment_duration, segment_start))
    
    # Если не найдено пауз, используем весь аудио как один сегмент
    if not speech_segments:
        speech_segments = [(audio_data, len(audio_data) / sample_rate, 0)]
    
    return speech_segments

def split_audio_by_silence_adaptive(audio_path, max_segment_duration=20):
    """
    Адаптивное разделение аудио по паузам с гарантированным разбиением
    """
    logger.info("Адаптивное разделение аудио...")
    
    # Загружаем аудио с оригинальной частотой дискретизации
    audio_data, original_sample_rate = librosa.load(audio_path, sr=None)
    logger.info(f"Исходная частота дискретизации: {original_sample_rate} Гц")
    
    # Предобработка аудио
    audio_data, sample_rate = preprocess_audio(audio_data, original_sample_rate)
    
    duration = len(audio_data) / sample_rate
    logger.info(f"Длительность аудио после предобработки: {duration:.2f} секунд")
    
    # Многоуровневый подход с разными параметрами
    silence_params = [
        {"min_silence_duration": 0.8, "silence_threshold": -35, "padding": 0.2},
        {"min_silence_duration": 0.5, "silence_threshold": -30, "padding": 0.1},
        {"min_silence_duration": 0.3, "silence_threshold": -25, "padding": 0.05},
    ]
    
    all_segments = []
    
    for i, params in enumerate(silence_params):
        logger.info(f"Попытка {i+1} с параметрами: пауза={params['min_silence_duration']}с, порог={params['silence_threshold']}dB")
        
        segments = _split_audio_data_by_silence(audio_data, sample_rate, **params)
        
        # Проверяем качество разделения
        if len(segments) > 1:
            logger.info(f"Успешно разделено на {len(segments)} сегментов")
            all_segments = segments
            break
        else:
            logger.info("Разделение не удалось, пробуем более чувствительные параметры")
    
    # Если все попытки не удались, используем весь аудиофайл как один сегмент
    if len(all_segments) <= 1:
        logger.info("Все попытки автоматического разделения не удались, используем весь файл как один сегмент")
        all_segments = [(audio_data, duration, 0)]
    
    # Рекурсивно обрабатываем слишком длинные сегменты
    final_segments = []
    for segment, seg_duration, start_time in all_segments:
        if seg_duration > max_segment_duration:
            logger.info(f"Сегмент слишком длинный ({seg_duration:.2f} сек), рекурсивное адаптивное разделение...")
            sub_segments = adaptive_split_segment(segment, sample_rate, start_time, max_segment_duration)
            final_segments.extend(sub_segments)
        else:
            final_segments.append((segment, seg_duration, start_time))
    
    logger.info(f"Финальное количество сегментов: {len(final_segments)}")
    
    # Проверяем, что все сегменты имеют разумную длительность
    max_found_duration = max([dur for _, dur, _ in final_segments]) if final_segments else 0
    if max_found_duration > max_segment_duration + 5:  # +5 для запаса
        logger.warning(f"Есть сегменты длиннее {max_segment_duration} сек: {max_found_duration:.2f} сек")
    
    return final_segments, sample_rate

def transcribe_audio_segment(audio_segment, sampling_rate):
    """Транскрибирует один сегмент аудио и возвращает текст"""
    try:
        # Проверяем длительность сегмента
        duration = len(audio_segment) / sampling_rate
        if duration < 0.1:  # Слишком короткий сегмент
            return ""
        
        # Ограничиваем максимальную длину для экономии памяти
        max_duration = 30  # секунд
        if duration > max_duration:
            logger.warning(f"Сегмент слишком длинный для обработки ({duration:.2f} сек)")
            return ""
        
        # Убеждаемся, что частота дискретизации правильная
        if sampling_rate != 16000:
            audio_segment = librosa.resample(audio_segment, orig_sr=sampling_rate, target_sr=16000)
            sampling_rate = 16000
        
        inputs = processor(audio_segment, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            logits = model(inputs.input_values).logits
            
        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_sentence = processor.batch_decode(predicted_ids)[0].strip()
        
        return predicted_sentence
    except Exception as e:
        logger.error(f"Ошибка транскрипции сегмента: {str(e)}")
        return ""

def process_audio_file(audio_path):
    """Обрабатывает аудиофайл, разделяя его по паузам и транскрибируя каждый сегмент"""
    try:
        logger.info(f"Обработка файла: {os.path.basename(audio_path)}")
        
        # Разделяем аудио на сегменты с гарантированным разбиением
        segments_info, sample_rate = split_audio_by_silence_adaptive(audio_path)
        
        results = []
        for i, (segment, duration, start_time) in enumerate(segments_info):
            logger.info(f"Транскрипция сегмента {i+1}/{len(segments_info)} (длительность: {duration:.2f} сек)...")
            text = transcribe_audio_segment(segment, sample_rate)
            if text:
                logger.info(f"Результат: {text[:80]}{'...' if len(text) > 80 else ''}")
            else:
                logger.warning("Результат: [пусто или ошибка]")
            results.append((text, duration, start_time))
        
        return results
            
    except Exception as e:
        logger.error(f"Ошибка обработки файла {audio_path}: {str(e)}")
        return [("", 0, 0)]

def process_audio_async(id, audio_path):
    """Асинхронная обработка аудио"""
    try:
        logger.info(f"Начата обработка для ID: {id}")
        logger.info(f"Аудиофайл: {audio_path}")
        
        # Проверяем существование файла
        if not os.path.exists(audio_path):
            logger.error(f"Файл {audio_path} не найден")
            return
        
        # Обрабатываем файл
        segment_results = process_audio_file(audio_path)
        
        scripts = []
        for text, duration, start_time in segment_results:
            if text:  # Добавляем только непустые результаты
                end_time = start_time + duration
                time_range = f"{format_timestamp(start_time)}-{format_timestamp(end_time)}"
                scripts.append(f"{time_range}: {text}")

        logger.info(f"Обработка завершена. Всего сегментов: {len(scripts)}")

        # Отправляем результат на callback URL
        callback_data = {
            "id": id,
            "scripts": scripts,
            "count": len(scripts)
        }
        
        callback_url = f"{CALLBACK_URL}/{id}"
        logger.info(f"Отправка callback на: {callback_url}")
        try:
            response = requests.post(callback_url, json=callback_data, timeout=30)
            logger.info(f"Callback отправлен, статус: {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка отправки callback: {str(e)}")
            
    except Exception as e:
        logger.error(f"Ошибка асинхронной обработки: {str(e)}")

@app.route('/api/v1/scripts/to-translate/<id>', methods=['POST'])
def transcribe_audio(id):
    data = request.get_json()
    if not data or 'path' not in data:
        logger.warning(f"Получен запрос без пути к файлу для ID: {id}")
        return jsonify({"error": "No path provided"}), 400
    
    audio_path = data['path']
    
    if not os.path.exists(audio_path) or not os.path.isfile(audio_path):
        logger.error(f"Аудиофайл не найден: {audio_path}")
        return jsonify({"error": f"Audio file not found: {audio_path}"}), 400
    
    # Проверяем расширение файла
    if not audio_path.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.ogg')):
        logger.error(f"Неподдерживаемый формат файла: {audio_path}")
        return jsonify({"error": f"Unsupported file format: {audio_path}"}), 400
    
    logger.info(f"Получен запрос на транскрипцию, ID: {id}")
    logger.info(f"Аудиофайл: {audio_path}")
    
    # Запускаем асинхронную обработку в отдельном потоке
    thread = threading.Thread(target=process_audio_async, args=(id, audio_path))
    thread.daemon = True
    thread.start()
    
    logger.info(f"Запущена асинхронная обработка для ID: {id}")
    return '', 200

@app.route('/health', methods=['GET'])
def health_check():
    logger.debug("Проверка здоровья сервера")
    return jsonify({"status": "healthy", "model": MODEL_ID})

@app.errorhandler(Exception)
def handle_exception(e):
    """Глобальный обработчик исключений"""
    logger.error(f"Необработанное исключение: {str(e)}", exc_info=True)
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    logger.info("Сервер транскрипции запускается на порту 9090")
    logger.info(f"Callback URL: {CALLBACK_URL}")
    logger.info(f"Используемая модель: {MODEL_ID}")
    app.run(host='0.0.0.0', port=9090, debug=False)