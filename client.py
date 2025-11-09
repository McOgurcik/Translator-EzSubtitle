import requests
import json
import sys
import os
import time
import tempfile
import librosa
import soundfile as sf
import numpy as np

#ЭТО ЭМУЛЯТОР РАЗБИЕНИЯ ДЛЯ ТЕСТОВ
# python client.py long_audio.wav СКРИПТ РАЗбИВАЕТ И ОТПРАВЛЯЕТ НА СЕРВЕР, А ЗАТЕМ ПРИНЕМАЕТ РЕЗУЛЬТАТ


def split_audio_file(audio_path, chunk_duration=10):
    """
    Разбивает аудиофайл на чанки заданной длительности (в секундах)
    Возвращает список путей к временным файлам с чанками
    """
    print(f"Загружаем и разбиваем аудиофайл: {audio_path}")
    
    # Загружаем аудио
    audio_data, sample_rate = librosa.load(audio_path, sr=None)
    duration = len(audio_data) / sample_rate
    
    print(f"Длительность аудио: {duration:.2f} секунд, частота: {sample_rate} Гц")
    
    # Рассчитываем размер чанка в сэмплах
    chunk_size = int(chunk_duration * sample_rate)
    
    # Создаем временные файлы для каждого чанка
    chunk_files = []
    
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i:i + chunk_size]
        chunk_duration_actual = len(chunk) / sample_rate
        
        # Пропускаем слишком короткие чанки (меньше 0.5 секунды)
        if chunk_duration_actual < 0.5:
            continue
            
        # Создаем временный файл для чанка
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            sf.write(tmp_file.name, chunk, sample_rate)
            chunk_files.append(tmp_file.name)
            
        print(f"Создан чанк {len(chunk_files)}: {chunk_duration_actual:.2f} секунд")
    
    print(f"Аудио разбито на {len(chunk_files)} чанков")
    return chunk_files

def send_audio_chunks_for_transcription(chunk_files, server_url="http://localhost:9090", audio_id="test_001"):
    """
    Отправляет несколько аудиофайлов на сервер транскрипции и возвращает результат
    """
    
    url = f"{server_url}/api/v1/scripts/to-translate/{audio_id}"
    
    try:
        # Подготавливаем файлы для отправки
        files = []
        for i, chunk_file in enumerate(chunk_files):
            files.append(('files', (f"chunk_{i:04d}.wav", open(chunk_file, 'rb'), 'audio/wav')))
        
        print(f"Отправка {len(chunk_files)} чанков на сервер...")
        start_time = time.time()
        response = requests.post(url, files=files, timeout=120)
        processing_time = time.time() - start_time
        
        # Закрываем файлы
        for _, file_data in files:
            file_data[1].close()
            
        if response.status_code == 200:
            result = response.json()
            print(f"Транскрипция успешно выполнена за {processing_time:.2f} секунд!")
            print(f"ID: {result.get('id')}")
            print(f"Количество фраз: {result.get('count')}")
            print(f"Общая длительность: {result.get('total_duration', 0):.2f} сек")
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
    finally:
        # Удаляем временные файлы чанков
        for chunk_file in chunk_files:
            try:
                os.unlink(chunk_file)
            except:
                pass

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

def main():
    if len(sys.argv) < 2:
        print("Использование: python client.py <путь_к_аудиофайлу> [chunk_duration] [server_url] [audio_id]")
        print("Пример: python client.py audio.wav 10 http://localhost:9090 test_001")
        print("  chunk_duration - длительность чанка в секундах (по умолчанию 10)")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    # Парсим аргументы
    chunk_duration = 10  # по умолчанию 10 секунд
    server_url = "http://localhost:9090"
    audio_id = "test_001"
    
    for i, arg in enumerate(sys.argv[2:], 2):
        if arg.isdigit():
            chunk_duration = int(arg)
        elif arg.startswith('http://') or arg.startswith('https://'):
            server_url = arg
        elif '=' in arg and arg.split('=')[0] == 'id':
            audio_id = arg.split('=')[1]
    
    if not os.path.exists(audio_file):
        print(f"Ошибка: файл {audio_file} не найден")
        sys.exit(1)
    
    # Проверяем здоровье сервера
    if not health_check(server_url):
        print("Сервер недоступен. Запустите server.py сначала.")
        sys.exit(1)
    
    # Разбиваем аудио на чанки
    chunk_files = split_audio_file(audio_file, chunk_duration)
    
    if not chunk_files:
        print("Не удалось разбить аудиофайл на чанки")
        sys.exit(1)
    
    # Отправляем чанки на транскрипцию
    result = send_audio_chunks_for_transcription(chunk_files, server_url, audio_id)
    
    if result is None:
        sys.exit(1)

if __name__ == '__main__':
    main()