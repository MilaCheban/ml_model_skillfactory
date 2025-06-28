import os
import shutil
import argparse
import logging

# настройка базовой конфигурации системы логирования
logging.basicConfig(
    # устанавливает уровень логирования на INFO (будет записывать сообщения уровня INFO и выше)
    level=logging.INFO, 
    # формат вывода: временная метка - уровень важности сообщения (INFO, ERROR и т.д.) - текст сообщения
    format='%(asctime)s - %(levelname)s - %(message)s')

def save_results(model_path, metrics_path, output_dir):
    '''
    Функция предназначена для сохранения результатов работы модели 
    (саму модель и метрики качества) в указанный каталог.
    Аргументы:
        model_path (str):   путь к файлу с обученной моделью (.pkl)
        metrics_path (str): путь к файлу с метриками модели (.json)
        output_dir (str):   каталог, куда будут сохранены результаты
    Функция возвращает:
        None: исключение при наличии ошибок
    '''
    # логирование начала операции
    logging.info("Сохранение результатов...")
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Копируем модель и метрики в выходную папку
        shutil.copy(model_path, os.path.join(output_dir, 'model.pkl'))
        shutil.copy(metrics_path, os.path.join(output_dir, 'metrics.json'))
        
        # логирование успешного завершения
        logging.info(f"Результаты сохранены в {output_dir}")
    except Exception as e:
        # логирует сообщение об ошибке с деталями исключения
        logging.error(f"Ошибка сохранения результатов: {e}")
        # повторно вызывает исключение (raise) для обработки на более высоком уровне
        raise

## Основной блок
if __name__ == "__main__":
    # cоздает парсер аргументов командной строки
    parser = argparse.ArgumentParser(description="Сохранение модели и метрик")
    # добавляет аргумент --model
    parser.add_argument('--model', type=str, default='model/model.pkl', help='Путь к модели')
    # добавляет аргумент --metrics
    parser.add_argument('--metrics', type=str, default='metrics/metrics.json', help='Путь к метрикам')
    # добавляет аргумент --output_dir
    parser.add_argument('--output_dir', type=str, default='results/', help='Выходная папка')
    # разбирает аргументы командной строки
    args = parser.parse_args()
    # вызывает функцию save_results, передавая путь из аргументов командной строки
    save_results(args.model, args.metrics, args.output_dir)