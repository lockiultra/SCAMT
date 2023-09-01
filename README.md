# SCAMT

Репозиторий для прохождения стажировки в компании SCAMT. Задача состоит в генерации новых лекарственных молекул и перепрофилировании существующих.

Для тренировки на ваших данных следует запустить следующий скрипт:
```
python train.py /path/to/data
```
Датасет должен быть в формате .csv и содержать следующие столбцы:
- Drug - название лекарственного препарата
- Smiles - представление молекулы в виде SMILES
- Disease - название болезни

Для генерации новых лекарственных молекул используиется натреннированная на датасете ZINC модель [DGMG](https://lifesci.dgl.ai/api/model.pretrain.htmls) из библиотеки dgllife. Для генерации лекарственных молекул следует запустить следующий скрипт:
```
python generate.py
```

Для проверки лекарственных свойств молекул следует запустить следующий скрипт:
```
python predict.py smiles
```

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)