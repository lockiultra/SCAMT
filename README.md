# SCAMT

Репозиторий для прохождения стажировки в компании SCAMT. Задача состоит в генерации новых лекарственных молекул и перепрофилировании существующих.

Для установки зависимостей следует запустить следующий скрипт:
```
pip install -r requirements.txt
```

Для тренировки на ваших данных следует запустить следующий скрипт:
```
python train.py /path/to/data
```
Датасет должен быть в формате .csv и содержать следующие столбцы:
- Drug - название лекарственного препарата
- Smiles - представление молекулы в виде SMILES
- Disease - название болезни

Для генерации новых лекарственных молекул используиется натреннированная на датасете ZINC модель [DGMG](https://lifesci.dgl.ai/api/model.pretrain.html) из библиотеки dgllife.
Модель DGMG имеет следующую структуру:
```
DGMG(
  (graph_embed): GraphEmbed(
    (node_gating): Sequential(
      (0): Linear(in_features=128, out_features=1, bias=True)
      (1): Sigmoid()
    )
    (node_to_graph): Linear(in_features=128, out_features=256, bias=True)
  )
  (graph_prop): GraphProp(
    (message_funcs): ModuleList(
      (0-1): 2 x Linear(in_features=259, out_features=256, bias=True)
    )
    (node_update_funcs): ModuleList(
      (0-1): 2 x GRUCell(256, 128)
    )
  )
  (add_node_agent): AddNode(
    (add_node): Sequential(
      (0): Linear(in_features=256, out_features=256, bias=True)
      (1): Dropout(p=0.2, inplace=False)
      (2): Linear(in_features=256, out_features=10, bias=True)
    )
    (node_type_embed): Embedding(9, 128)
    (initialize_hv): Linear(in_features=384, out_features=128, bias=True)
    (dropout): Dropout(p=0.2, inplace=False)
  )
  (add_edge_agent): AddEdge(
    (add_edge): Sequential(
      (0): Linear(in_features=384, out_features=384, bias=True)
      (1): Dropout(p=0.2, inplace=False)
      (2): Linear(in_features=384, out_features=4, bias=True)
    )
  )
  (choose_dest_agent): ChooseDestAndUpdate(
    (choose_dest): Sequential(
      (0): Linear(in_features=259, out_features=259, bias=True)
      (1): Dropout(p=0.2, inplace=False)
      (2): Linear(in_features=259, out_features=1, bias=True)
    )
  )
) 
```
Для генерации лекарственных молекул следует запустить следующий скрипт:
```
python generate.py
```

Для проверки лекарственных свойств молекул следует запустить следующий скрипт:
```
python predict.py smiles
```

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)