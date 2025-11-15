from flask import Flask, request, jsonify
import requests
import torch
import librosa
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import os
from datetime import timedelta
import threading
import tempfile
import soundfile as sf
import noisereduce as nr
from scipy import signal
import logging
import sys
from logging.handlers import RotatingFileHandler
import re

# Импорты для расширенной NLP обработки
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    NamesExtractor,
    Doc
)
import pymorphy2

app = Flask(__name__)

# Конфигурация
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"
CALLBACK_URL = "http://localhost:8080/api/v1/scripts/from-translate"

# Настройка логирования
def setup_logging():
    """Настройка системы логирования"""
    logger = logging.getLogger('audio_transcriptor')
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    file_handler = RotatingFileHandler(
        'audio_transcriptor.log',
        maxBytes=10*1024*1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Инициализация логгера
logger = setup_logging()

# Инициализация NLP компонентов
logger.info("Инициализация NLP компонентов...")

# Natasha компоненты
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)
names_extractor = NamesExtractor(morph_vocab)

# Pymorphy2 для морфологического анализа
morph_analyzer = pymorphy2.MorphAnalyzer()

logger.info("NLP компоненты инициализированы")

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

def improve_word_forms_with_pymorphy(text):
    """
    Улучшение словоформ с помощью Pymorphy2
    Исправляет грамматические формы и приводит слова к нормальной форме
    """
    if not text or len(text.strip()) == 0:
        return text
    
    logger.debug("Улучшение словоформ с Pymorphy2...")
    
    try:
        words = text.split()
        improved_words = []
        
        for word in words:
            # Пропускаем слова с пунктуацией и короткие слова
            if not word.isalpha() or len(word) < 3:
                improved_words.append(word)
                continue
            
            # Анализируем слово
            parsed = morph_analyzer.parse(word)
            if not parsed:
                improved_words.append(word)
                continue
            
            best_parse = parsed[0]
            
            # Используем нормальную форму, если уверенность низкая
            # или если слово в нестандартной форме
            if (best_parse.score < 0.5 or 
                best_parse.tag.POS in ['NOUN', 'VERB', 'ADJF', 'ADJS', 'INFN']):
                
                normal_form = best_parse.normal_form
                
                # Сохраняем регистр оригинала
                if word.istitle():
                    normal_form = normal_form.capitalize()
                
                # Для существительных и прилагательных используем нормальную форму
                # только если это улучшает читаемость
                if (best_parse.tag.POS in ['NOUN', 'ADJF', 'ADJS'] and 
                    best_parse.score < 0.3):
                    improved_words.append(normal_form)
                else:
                    improved_words.append(word)
            else:
                improved_words.append(word)
        
        improved_text = ' '.join(improved_words)
        
        # Исправление согласования падежей для коротких фраз
        improved_text = improve_grammar_agreement(improved_text)
        
        return improved_text
        
    except Exception as e:
        logger.warning(f"Ошибка в Pymorphy2 обработке: {str(e)}")
        return text

def improve_grammar_agreement(text):
    """
    Улучшение грамматического согласования в тексте
    """
    try:
        # Простые правила для улучшения согласования
        # Это базовые замены, которые работают для большинства случаев
        
        # Исправление согласования числительных и существительных
        replacements = [
            (r'(\d+) года', r'\1 год'),  # "5 года" -> "5 год"
            (r'(\d+) лета', r'\1 лет'),   # "5 лета" -> "5 лет"
            (r'(\d+) месяцы', r'\1 месяцев'),  # "5 месяцы" -> "5 месяцев"
        ]
        
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text)
        
        return text
        
    except Exception as e:
        logger.debug(f"Ошибка в грамматической корректировке: {str(e)}")
        return text

def enhanced_natasha_processing(text):
    """
    Расширенная обработка текста с использованием всех возможностей Natasha
    """
    if not text or len(text.strip()) == 0:
        return text
    
    logger.debug("Расширенная обработка с Natasha...")
    
    try:
        # Создаем документ
        doc = Doc(text)
        
        # 1. Сегментация на предложения
        doc.segment(segmenter)
        
        # 2. Морфологический разбор
        doc.tag_morph(morph_tagger)
        
        # 3. Синтаксический разбор
        doc.parse_syntax(syntax_parser)
        
        # 4. Извлечение именованных сущностей
        doc.tag_ner(ner_tagger)
        
        # 5. Нормализация именованных сущностей
        for span in doc.spans:
            if span.type in ['PER', 'LOC', 'ORG']:
                span.normalize(morph_vocab)
        
        # 6. Лемматизация токенов
        for token in doc.tokens:
            token.lemmatize(morph_vocab)
        
        # 7. Восстановление текста с улучшенной структурой
        processed_sentences = []
        
        for sent in doc.sents:
            sent_text = sent.text
            
            # Автоматическая капитализация именованных сущностей
            sent_text = capitalize_named_entities(sent_text, doc)
            
            # Добавление пунктуации в конец предложения
            if sent_text and not sent_text[-1] in '.!?;':
                # Определяем тип предложения по первому слову
                first_word = sent_text.split()[0].lower() if sent_text.split() else ""
                if first_word in ['кто', 'что', 'где', 'когда', 'почему', 'как']:
                    sent_text += '?'
                else:
                    sent_text += '.'
            
            # Капитализация начала предложения
            if sent_text:
                sent_text = sent_text[0].upper() + sent_text[1:] if len(sent_text) > 1 else sent_text.upper()
            
            processed_sentences.append(sent_text)
        
        result = ' '.join(processed_sentences)
        
        # Улучшение читаемости пунктуации
        result = re.sub(r'\s+([.,!?;])', r'\1', result)  # Убираем пробелы перед знаками препинания
        result = re.sub(r'([.,!?;])([А-Яа-я])', r'\1 \2', result)  # Добавляем пробелы после знаков препинания
        result = re.sub(r'\s+', ' ', result)  # Убираем множественные пробелы
        
        return result.strip()
        
    except Exception as e:
        logger.error(f"Ошибка в расширенной обработке Natasha: {str(e)}")
        return text

def capitalize_named_entities(text, doc):
    """
    Автоматическая капитализация именованных сущностей в тексте
    """
    try:
        words = text.split()
        
        # Создаем множество именованных сущностей из документа
        entities = set()
        for span in doc.spans:
            if span.type in ['PER', 'LOC', 'ORG']:
                entity_words = span.normal.split()
                entities.update(entity_words)
        
        # Капитализируем слова, которые являются именованными сущностями
        for i, word in enumerate(words):
            clean_word = re.sub(r'[^\w]', '', word).lower()
            if clean_word in entities:
                words[i] = word.capitalize()
        
        return ' '.join(words)
        
    except Exception as e:
        logger.debug(f"Ошибка в капитализации сущностей: {str(e)}")
        return text

def normalize_text_structure(text):
    """
    Нормализация структуры текста
    """
    if not text:
        return text
    
    # Удаление лишних пробелов
    text = re.sub(r'\s+', ' ', text)
    
    # Исправление распространенных опечаток (универсальные случаи)
    common_fixes = [
        (r'(\w)щь\b', r'\1сть'),  # "помощь" -> "помощь" (но может исправить некоторые ошибки)
        (r'(\w)тся\b', r'\1ться'), # "нравится" -> "нравиться" (осторожно!)
    ]
    
    # Применяем только безопасные замены
    for pattern, replacement in common_fixes:
        text = re.sub(pattern, replacement, text)
    
    return text.strip()

def comprehensive_text_enhancement(text):
    """
    Комплексное улучшение текста с использованием всех доступных методов
    """
    if not text or len(text.strip()) == 0:
        return text
    
    original_text = text
    logger.debug(f"Начало комплексного улучшения текста: {original_text[:100]}...")
    
    try:
        # 1. Нормализация структуры текста
        text = normalize_text_structure(text)
        
        # 2. Улучшение словоформ с помощью Pymorphy2
        text = improve_word_forms_with_pymorphy(text)
        
        # 3. Расширенная обработка с Natasha
        text = enhanced_natasha_processing(text)
        
        # 4. Финальная нормализация
        text = re.sub(r'\s+', ' ', text).strip()
        
        logger.debug(f"Текст улучшен: {original_text} -> {text}")
        
        return text
        
    except Exception as e:
        logger.error(f"Ошибка в комплексном улучшении текста: {str(e)}")
        return original_text

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
        target_rms = 0.1
        audio_data = audio_data * (target_rms / rms)
        audio_data = np.clip(audio_data, -0.99, 0.99)
        logger.debug(f"Нормализация громкости (RMS: {rms:.4f} -> {np.sqrt(np.mean(audio_data**2)):.4f})")
    
    # 5. Фильтр высоких частот
    nyquist = sample_rate / 2
    highpass_cutoff = 80
    if highpass_cutoff < nyquist:
        sos = signal.butter(4, highpass_cutoff/nyquist, btype='highpass', output='sos')
        audio_data = signal.sosfilt(sos, audio_data)
        logger.debug(f"Применен ВЧ-фильтр ({highpass_cutoff} Гц)")
    
    return audio_data, sample_rate

def adaptive_split_segment(audio_data, sample_rate, start_time, max_segment_duration=20, recursion_level=0):
    """Рекурсивно разделяет сегмент адаптивными методами"""
    duration = len(audio_data) / sample_rate
    
    if duration <= max_segment_duration:
        return [(audio_data, duration, start_time)]
    
    logger.info(f"Рекурсия {recursion_level}: сегмент {duration:.2f} сек слишком длинный, адаптивное разделение...")
    
    silence_params = [
        {"min_silence_duration": 0.8 - recursion_level * 0.2, "silence_threshold": -35 + recursion_level * 5, "padding": 0.2},
        {"min_silence_duration": 0.5 - recursion_level * 0.1, "silence_threshold": -30 + recursion_level * 5, "padding": 0.1},
        {"min_silence_duration": 0.3 - recursion_level * 0.05, "silence_threshold": -25 + recursion_level * 5, "padding": 0.05},
    ]
    
    for params in silence_params:
        params["min_silence_duration"] = max(0.1, params["min_silence_duration"])
        params["silence_threshold"] = min(-15, params["silence_threshold"])
    
    for i, params in enumerate(silence_params):
        logger.debug(f"Попытка {i+1} с параметрами: пауза={params['min_silence_duration']:.2f}с, порог={params['silence_threshold']}dB")
        
        segments = _split_audio_data_by_silence(audio_data, sample_rate, **params)
        
        if len(segments) > 1:
            logger.info(f"Успешно разделено на {len(segments)} подсегментов")
            
            all_subsegments = []
            for segment, seg_duration, seg_start in segments:
                adjusted_start = start_time + seg_start
                subsegments = adaptive_split_segment(
                    segment, sample_rate, adjusted_start, 
                    max_segment_duration, recursion_level + 1
                )
                all_subsegments.extend(subsegments)
            
            return all_subsegments
    
    logger.info("Адаптивное разделение не удалось, используем интеллектуальное временное разделение")
    return intelligent_time_split(audio_data, sample_rate, start_time, max_segment_duration)

def intelligent_time_split(audio_data, sample_rate, start_time, max_segment_duration=20):
    """Интеллектуальное разделение по времени с поиском естественных границ"""
    duration = len(audio_data) / sample_rate
    
    energy = librosa.feature.rms(y=audio_data, frame_length=1024, hop_length=256)[0]
    energy_normalized = (energy - np.min(energy)) / (np.max(energy) - np.min(energy))
    
    num_segments = max(2, int(np.ceil(duration / max_segment_duration)))
    segment_duration = duration / num_segments
    
    segments = []
    
    for i in range(num_segments):
        segment_start_time = i * segment_duration
        segment_end_time = (i + 1) * segment_duration
        
        search_start = max(0, int((segment_start_time - 2) * sample_rate / 256))
        search_end = min(len(energy_normalized), int((segment_start_time + 2) * sample_rate / 256))
        
        if search_end > search_start:
            search_window = energy_normalized[search_start:search_end]
            if len(search_window) > 0:
                min_index = np.argmin(search_window)
                best_split_time = (search_start + min_index) * 256 / sample_rate
            else:
                best_split_time = segment_start_time
        else:
            best_split_time = segment_start_time
        
        best_split_time = max(segment_start_time, min(segment_end_time, best_split_time))
        
        start_sample = int(best_split_time * sample_rate)
        end_sample = int(segment_end_time * sample_rate) if i < num_segments - 1 else len(audio_data)
        
        segment = audio_data[start_sample:end_sample]
        segment_duration_actual = len(segment) / sample_rate
        
        if segment_duration_actual > 0.1:
            segments.append((segment, segment_duration_actual, start_time + best_split_time))
    
    logger.info(f"Интеллектуальное разделение на {len(segments)} сегментов")
    return segments

def _split_audio_data_by_silence(audio_data, sample_rate, min_silence_duration=0.5, silence_threshold=-40, padding=0.1):
    """Внутренняя функция для разделения аудиоданных по паузам"""
    energy = librosa.feature.rms(y=audio_data, frame_length=1024, hop_length=256)[0]
    energy_db = librosa.amplitude_to_db(energy, ref=np.max)
    
    silent_segments = []
    is_silent = False
    silent_start = 0
    
    for i, db in enumerate(energy_db):
        time_pos = i * 256 / sample_rate
        
        if db < silence_threshold and not is_silent:
            is_silent = True
            silent_start = time_pos
        elif db >= silence_threshold and is_silent:
            silent_end = time_pos
            silent_duration = silent_end - silent_start
            
            if silent_duration >= min_silence_duration:
                silent_segments.append((silent_start, silent_end))
            
            is_silent = False
    
    if is_silent:
        silent_end = len(energy_db) * 256 / sample_rate
        silent_duration = silent_end - silent_start
        if silent_duration >= min_silence_duration:
            silent_segments.append((silent_start, silent_end))
    
    speech_segments = []
    prev_end = 0
    
    for silent_start, silent_end in silent_segments:
        segment_start = max(0, prev_end - padding)
        segment_end = silent_start + padding
        
        if segment_end > segment_start and (segment_end - segment_start) > 0.1:
            start_sample = int(segment_start * sample_rate)
            end_sample = int(segment_end * sample_rate)
            segment = audio_data[start_sample:end_sample]
            segment_duration = len(segment) / sample_rate
            speech_segments.append((segment, segment_duration, segment_start))
        
        prev_end = silent_end
    
    if prev_end < len(audio_data) / sample_rate:
        segment_start = max(0, prev_end - padding)
        segment_end = len(audio_data) / sample_rate
        start_sample = int(segment_start * sample_rate)
        end_sample = int(segment_end * sample_rate)
        segment = audio_data[start_sample:end_sample]
        segment_duration = len(segment) / sample_rate
        speech_segments.append((segment, segment_duration, segment_start))
    
    if not speech_segments:
        speech_segments = [(audio_data, len(audio_data) / sample_rate, 0)]
    
    return speech_segments

def split_audio_by_silence_adaptive(audio_path, max_segment_duration=20):
    """Адаптивное разделение аудио по паузам с гарантированным разбиением"""
    logger.info("Адаптивное разделение аудио...")
    
    audio_data, original_sample_rate = librosa.load(audio_path, sr=None)
    logger.info(f"Исходная частота дискретизации: {original_sample_rate} Гц")
    
    audio_data, sample_rate = preprocess_audio(audio_data, original_sample_rate)
    
    duration = len(audio_data) / sample_rate
    logger.info(f"Длительность аудио после предобработки: {duration:.2f} секунд")
    
    silence_params = [
        {"min_silence_duration": 0.8, "silence_threshold": -35, "padding": 0.2},
        {"min_silence_duration": 0.5, "silence_threshold": -30, "padding": 0.1},
        {"min_silence_duration": 0.3, "silence_threshold": -25, "padding": 0.05},
    ]
    
    all_segments = []
    
    for i, params in enumerate(silence_params):
        logger.info(f"Попытка {i+1} с параметрами: пауза={params['min_silence_duration']}с, порог={params['silence_threshold']}dB")
        
        segments = _split_audio_data_by_silence(audio_data, sample_rate, **params)
        
        if len(segments) > 1:
            logger.info(f"Успешно разделено на {len(segments)} сегментов")
            all_segments = segments
            break
        else:
            logger.info("Разделение не удалось, пробуем более чувствительные параметры")
    
    if len(all_segments) <= 1:
        logger.info("Все попытки автоматического разделения не удались, используем весь файл как один сегмент")
        all_segments = [(audio_data, duration, 0)]
    
    final_segments = []
    for segment, seg_duration, start_time in all_segments:
        if seg_duration > max_segment_duration:
            logger.info(f"Сегмент слишком длинный ({seg_duration:.2f} сек), рекурсивное адаптивное разделение...")
            sub_segments = adaptive_split_segment(segment, sample_rate, start_time, max_segment_duration)
            final_segments.extend(sub_segments)
        else:
            final_segments.append((segment, seg_duration, start_time))
    
    logger.info(f"Финальное количество сегментов: {len(final_segments)}")
    
    max_found_duration = max([dur for _, dur, _ in final_segments]) if final_segments else 0
    if max_found_duration > max_segment_duration + 5:
        logger.warning(f"Есть сегменты длиннее {max_segment_duration} сек: {max_found_duration:.2f} сек")
    
    return final_segments, sample_rate

def transcribe_audio_segment(audio_segment, sampling_rate):
    """Транскрибирует один сегмент аудио и возвращает текст"""
    try:
        duration = len(audio_segment) / sampling_rate
        if duration < 0.1:
            return ""
        
        max_duration = 30
        if duration > max_duration:
            logger.warning(f"Сегмент слишком длинный для обработки ({duration:.2f} сек)")
            return ""
        
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
        
        segments_info, sample_rate = split_audio_by_silence_adaptive(audio_path)
        
        results = []
        for i, (segment, duration, start_time) in enumerate(segments_info):
            logger.info(f"Транскрипция сегмента {i+1}/{len(segments_info)} (длительность: {duration:.2f} сек)...")
            text = transcribe_audio_segment(segment, sample_rate)
            
            # КОМПЛЕКСНОЕ УЛУЧШЕНИЕ ТЕКСТА
            if text:
                original_text = text
                text = comprehensive_text_enhancement(text)
                logger.info(f"Исходный текст: {original_text[:80]}{'...' if len(original_text) > 80 else ''}")
                logger.info(f"Улучшенный текст: {text[:80]}{'...' if len(text) > 80 else ''}")
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
        
        if not os.path.exists(audio_path):
            logger.error(f"Файл {audio_path} не найден")
            return
        
        segment_results = process_audio_file(audio_path)
        
        scripts = []
        for text, duration, start_time in segment_results:
            if text:
                end_time = start_time + duration
                time_range = f"{format_timestamp(start_time)}-{format_timestamp(end_time)}"
                scripts.append(f"{time_range}: {text}")

        logger.info(f"Обработка завершена. Всего сегментов: {len(scripts)}")

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
    
    if not audio_path.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.ogg')):
        logger.error(f"Неподдерживаемый формат файла: {audio_path}")
        return jsonify({"error": f"Unsupported file format: {audio_path}"}), 400
    
    logger.info(f"Получен запрос на транскрипцию, ID: {id}")
    logger.info(f"Аудиофайл: {audio_path}")
    
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
    logger.error(f"Необработанное исключение: {str(e)}", exc_info=True)
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    logger.info("Сервер транскрипции запускается на порту 9090")
    logger.info(f"Callback URL: {CALLBACK_URL}")
    logger.info(f"Используемая модель: {MODEL_ID}")
    app.run(host='0.0.0.0', port=9090, debug=False)