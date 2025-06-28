import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import json
import argparse
import logging

# настройка базовой конфигурации системы логирования
logging.basicConfig(
    # устанавливает уровень логирования на INFO (будет записывать сообщения уровня INFO и выше)
    level=logging.INFO, 
    # формат вывода: временная метка - уровень важности сообщения (INFO, ERROR и т.д.) - текст сообщения
    format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_model(input_path, model_path, output_path):
    '''
    функция для оценки качества модели
    Аргументы:
        input_path (str): путь к тестовым данным (csv)
        model_path (str): путь к сохраненной модели (pickle-файл)
        output_path (str): путь для сохранения метрик (JSON-файл)
    Функция возвращает:
        None: исключение при наличии ошибок
    '''
    # логирование начала процесса
    logging.info("Оценка модели...")
    try:
        df = pd.read_csv(input_path)
        X = df.drop('diagnosis', axis=1)
        y = df['diagnosis']
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        y_pred = model.predict(X)
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred)
        }
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        # логирование успешного завершения
        logging.info(f"Метрики сохранены в {output_path}")
        return None
    except Exception as e:
        # логирует сообщение об ошибке с деталями исключения
        logging.error(f"Ошибка оценки: {e}")
        # повторно вызывает исключение (raise) для обработки на более высоком уровне
        raise

## Основной блок
if __name__ == "__main__":
    # cоздает парсер аргументов командной строки
    parser = argparse.ArgumentParser(description="Оценка модели")
    # добавляет аргумент --intput
    parser.add_argument('--input', type=str, default='data/processed_data_test.csv', help='Путь к test-датасету')
    # добавляет аргумент --model
    parser.add_argument('--model', type=str, default='model/model.pkl', help='Путь к модели')
    # добавляет аргумент --output
    parser.add_argument('--output', type=str, default='metrics/metrics.json', help='Путь для сохранения метрик')
    # разбирает аргументы командной строки
    args = parser.parse_args()
    # вызывает функцию evaluate_model, передавая путь из аргументов командной строки
    evaluate_model(args.input, args.model, args.output)