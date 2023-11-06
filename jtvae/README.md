# Процесс обучения JTVAE

1. Генерация vocab
```
python fast_jtnn/mol_tree.py -i ./data/smiles.txt -v ./data/vocab.txt
```

2. Предварительная обработка данных

```
python preprocess.py --train ./data/smiles.txt --split 100 --jobs
```

3. Обучение JTVAE
```
python vae_train.py --train processed --vocab ./data/vocab.txt --save_dir vae_model/
```