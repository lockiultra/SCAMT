# SCAMT

Репозиторий для прохождения стажировки в компании SCAMT. Задача состоит в генерации новых лекарственных молекул и перепрофилировании существующих.

### Классификатор

В ходе работы были обучены 8 графовых моделей на основе Message Passing Neural Network (MPNN) для классификации лекарственных молекул на следующие типвы болезней:
- Сердечно-сосудистые заболевания (Cardiovascular diseases)
- Заболевания пищеварительной системы (Digestive system diseases)
- Психические и поведенческие расстройства (Mental and behavioural disorders)
- Заболевания метаболической системы (Metabloic diseases)
- Заболевания нервной системы (Nervous system diseases)
- Заболевания кожных покровов (Skin and connective tissue diseases)
- Заболевания мочеывделительной системы (Urinary system diseases)

Обученные модели для каждого типа расположены в папке Models под видом {disease}.h5. Модели обучены на основе датасета (./data/data.csv) собранного на сайте [ClinicalTrials](https://clinicaltrials.gov)

В ходе обучения были получены слежующие результаты для каждой модели:
| Model     | Train AUC | Val AUC |
|----------|----------|----------|
| Cardiovascular diseases     | 0.8688   | 0.8515 |
| Digestive system diseases    | 0.7973   | 0.8421 |
| Mental and behavioural disorders   | 0.9423   | 0.9957
| Metabolic diseases   | 0.8163   | 0.8498 |
| Nervous system diseases   | 0.8962   | 0.9159 |
| Skin and connective tissue diseases   | 0.9753  | 0.9945
| Immune system diseases   | 0.7658  | 0.8020 |
| Urinary system diseases   | 0.8492  | 0.9549 |

### Генератор

В качестве модели для генерации лекарственных молекул был выбран Junction Tree VAE (JTVAE). Код модели содержится в папке jtvae.

Обученная модель находится в папке Models под названием model.epoch-19


[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
