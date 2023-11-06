import pandas as pd
import pickle
import sys
from DiseasePipeline import DiseasePipeline

def train(path_to_csv):
    data = pd.read_csv(path_to_csv)
    disease_pipeline = DiseasePipeline(data)
    disease_pipeline.train()
    for model in disease_pipeline.models:
        model.save(f'./Models/model_{model}.h5')

if __name__ == '__main__':
    train(sys.argv[1])