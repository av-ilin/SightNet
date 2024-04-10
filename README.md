# SightNet

[![License](https://img.shields.io/github/license/av-ilin/SightNet)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/downloads/)
[![TensorFlow Version](https://img.shields.io/badge/tensorflow-2.x-orange)](https://www.tensorflow.org/)
[![google-api-python-client](https://img.shields.io/badge/google--api--python--client-2.33.0-blue)](https://pypi.org/project/google-api-python-client/)
[![imgaug](https://img.shields.io/badge/imgaug-0.4.0-blue)](https://pypi.org/project/imgaug/)
[![tqdm](https://img.shields.io/badge/tqdm-4.62.3-green)](https://pypi.org/project/tqdm/)
[![opencv-python](https://img.shields.io/badge/opencv--python-4.5.3.56-green)](https://pypi.org/project/opencv-python/)


## Основные функции

- Аугментация изображений
- Детектирование поз с использованием модели PoseNet
- Создание датасета с ключевыми точками

## Установка

1. Установите необходимые зависимости, выполнив команду:

```bash
pip install -r packages.txt
```

## Конфигурация

Вы можете настроить папки и пути к моделям, используя аргументы командной строки. Доступные параметры:

- `--input_folder`: папка с исходными изображениями (по умолчанию: `src/data`)
- `--output_folder_augmented`: папка для сохранения аугментированных изображений (по умолчанию: `src/output/images/augmented`)
- `--augment_count`: количество аугментированных копий для каждого изображения (по умолчанию: `5`)
- `--output_folder_pose_detected`: папка для сохранения изображений с обнаруженными позами (по умолчанию: `src/output/images/pose_detected`)
- `--model_path`: путь к модели PoseNet (по умолчанию: `src/models/PoseNet.tflite`)
- `--csv_path`: путь для сохранения CSV файла с ключевыми точками (по умолчанию: `src/output/csv/keypoints.csv`)

Пример использования с аргументами:

```bash
python src\index.py --input_folder your_input_folder --output_folder_augmented your_output_folder_augmented --augment_count 5 --output_folder_pose_detected your_output_folder_pose_detected --model_path your_model_path --csv_path your_csv_path
```

## Лицензия

Этот проект лицензируется под MIT License. Подробности можно найти в файле `LICENSE`.

---