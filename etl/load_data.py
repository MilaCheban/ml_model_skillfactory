import pandas as pd
import requests
import argparse
import logging

# настройка базовой конфигурации системы логирования
logging.basicConfig(
    # устанавливает уровень логирования на INFO (будет записывать сообщения уровня INFO и выше)
    level=logging.INFO, 
    # формат вывода: временная метка - уровень важности сообщения (INFO, ERROR и т.д.) - текст сообщения
    format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(output_path):
    '''
    Функция для загрузки датасета Breast Cancer Wisconsin (Diagnostic).
    Аргументы:
        output_path (str): путь для сохранения файла
    Функция возвращает:
        None: исключение при наличии ошибок
    '''
    # логирование начала загрузки
    logging.info("Загрузка датасета...")
    # URL датасета
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    try:
        # выполняет HTTP-запрос для получения данных
        response = requests.get(url)
        # вызывает исключение, если HTTP-запрос завершился с ошибкой
        response.raise_for_status()
        # записывает содержимое ответа (response.text) в файл
        with open(output_path, 'w') as f:
            f.write(response.text)
        df = pd.read_csv(output_path, header=None)
        # логирование об успешной загрузке
        logging.info(f"Датасет загружен и сохранён в {output_path}")
        return None
    # обработка ошибок
    except Exception as e:
        # логирует сообщение об ошибке с деталями исключения
        logging.error(f"Ошибка загрузки данных: {e}")
        # повторно вызывает исключение (raise) для обработки на более высоком уровне
        raise
    
def load_data_local(input_path = 'data/wdbc.data', output_path = 'data/data.csv'):
    '''
    Функция для локальной загрузки датасета Breast Cancer Wisconsin (Diagnostic).
    Аргументы:
        output_path (str): путь исходного файла с данными
        output_path (str): путь для сохранения файла
    Функция возвращает:
        None: исключение при наличии ошибок
    '''
    # логирование начала загрузки
    logging.info("Загрузка датасета из локального источника...")
    try:
        df = pd.read_csv(input_path, header=None)
        df.to_csv(output_path, index=False)
        # логирование об успешной загрузке
        logging.info(f"Датасет загружен и сохранён в {output_path}")
        return None
    # обработка ошибок
    except Exception as e:
        # логирует сообщение об ошибке с деталями исключения
        logging.error(f"Ошибка загрузки данных: {e}")
        # повторно вызывает исключение (raise) для обработки на более высоком уровне
        raise

## Основной блок
if __name__ == "__main__":
    # cоздает парсер аргументов командной строки
    parser = argparse.ArgumentParser(description="Загрузка датасета Breast Cancer")
    # добавляет аргумент --output
    parser.add_argument('--output', type=str, default='data/data.csv', help='Путь для сохранения файла')
    # разбирает аргументы командной строки
    args = parser.parse_args()
    # вызывает функцию load_data, передавая путь из аргументов командной строки
    load_data(args.output)