import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import argparse
import logging

# настройка базовой конфигурации системы логирования
logging.basicConfig(
    # устанавливает уровень логирования на INFO (будет записывать сообщения уровня INFO и выше)
    level=logging.INFO, 
    # формат вывода: временная метка - уровень важности сообщения (INFO, ERROR и т.д.) - текст сообщения
    format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_data(input_path, output_path_train, output_path_test):
    '''
    Функция для предобработки данных.
    Аргументы:
        input_path: путь к исходному сsv-файлу 
        output_path_train: путь для сохранения обработанных данных (train-датасет)
        output_path_test: путь для сохранения обработанных данных (test-датасет)
    Функция возвращает:
        None: исключение при наличии ошибок
    '''
    # записывает информационное сообщение о начале предобработки
    logging.info("Предобработка данных...")
    try:
        # определяем имена столбцов
        columns = ['id', 'diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
        df = pd.read_csv(input_path,  header=None, skiprows=1, names=columns)
        
        # удаляем столбец ID
        df = df.drop('id', axis=1)
        
        # обрабатываем пропущенные значения
        df = df.dropna()
        
        # кодируем метки ('M' (злокачественная) → 1, 'B' (доброкачественная) → 0)
        le = LabelEncoder()
        df['diagnosis'] = le.fit_transform(df['diagnosis'])
        
        # разбиваем выборку на train/test со стратификацией
        train_df, test_df = train_test_split(df, test_size=0.2, 
                                   stratify=df['diagnosis'], 
                                   random_state=42)
        
        # нормализуем признаки
        scaler = StandardScaler()
        feature_cols = [col for col in df.columns if col != 'diagnosis']
        train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
        test_df[feature_cols] = scaler.transform(test_df[feature_cols])
        
        ## сохраняем обработанные данные
        train_df.to_csv(output_path_train, index=False)
        test_df.to_csv(output_path_test, index=False)
        # логирует сообщение об успешном завершении
        logging.info(f"Обработанные данные сохранены в {output_path_train, output_path_test}")
        return None
    except Exception as e:
        # логирует сообщение об ошибке с деталями исключения
        logging.error(f"Ошибка предобработки: {e}")
        # повторно вызывает исключение для обработки на более высоком уровне
        raise

## Основной блок
if __name__ == "__main__":
    # cоздает парсер аргументов командной строки
    parser = argparse.ArgumentParser(description="Предобработка датасета Breast Cancer")
    # добавляет аргумент --intput
    parser.add_argument('--input', type=str, default='data/data.csv', help='Путь к входному файлу')
    # добавляет аргумент --output_train
    parser.add_argument('--output_train', type=str, default='data/processed_data_train.csv', help='Путь для сохранения train-датасета')
    # добавляет аргумент --output_test
    parser.add_argument('--output_test', type=str, default='data/processed_data_test.csv', help='Путь для сохранения test-датасета')
    # разбирает аргументы командной строки
    args = parser.parse_args()
    # вызывает функцию preprocess_data, передавая путь из аргументов командной строки
    preprocess_data(args.input, args.output_train, args.output_test)