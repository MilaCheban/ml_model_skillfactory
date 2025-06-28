import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import argparse
import logging

# настройка базовой конфигурации системы логирования
logging.basicConfig(
    # устанавливает уровень логирования на INFO (будет записывать сообщения уровня INFO и выше)
    level=logging.INFO, 
    # формат вывода: временная метка - уровень важности сообщения (INFO, ERROR и т.д.) - текст сообщения
    format='%(asctime)s - %(levelname)s - %(message)s')

def train_model(input_path, output_path):
    '''
    Функция обучения модели.
    Аргументы:
        input_path (str):  путь к предобработанным данным (train-датасет)
        output_path (str): путь для сохранения обученной модели
    Функция возвращает:
        None: исключение при наличии ошибок
    '''
    # логирование начала процесса
    logging.info("Обучение модели...")
    try:
        df = pd.read_csv(input_path)
        X = df.drop('diagnosis', axis=1)
        y = df['diagnosis']
        
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
        # логирует сообщение об успешном сохранении модели
        logging.info(f"Модель сохранена в {output_path}")
        return None
    except Exception as e:
        # логирует сообщение об ошибке с деталями исключения
        logging.error(f"Ошибка обучения модели: {e}")
        # повторно вызывает исключение для обработки на более высоком уровне
        raise

## Основной блок
if __name__ == "__main__":
    # cоздает парсер аргументов командной строки
    parser = argparse.ArgumentParser(description="Обучение модели логистической регрессии")
    # добавляет аргумент --intput
    parser.add_argument('--input', type=str, default='data/processed_data_train.csv', help='Путь к train-датасету')
    # добавляет аргумент --output
    parser.add_argument('--output', type=str, default='model/model.pkl', help='Путь для сохранения модели')
    # разбирает аргументы командной строки
    args = parser.parse_args()
    # вызывает функцию train_model, передавая путь из аргументов командной строки
    train_model(args.input, args.output)