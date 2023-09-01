import pandas as pd
import pickle
import sys
from DiseasePipeline import DiseasePipeline

def train(path_to_csv):
    data = pd.read_csv(path_to_csv)
    disease_pipeline = DiseasePipeline(data)
    disease_pipeline.train()
    # for model in disease_pipeline.models:
    #     with open(f'./Models/{model}', 'wb') as f:
    #         pickle.dump(disease_pipeline.models[model], f)
    with open('./Models/disease_pipeline', 'wb') as f:
        pickle.dump(disease_pipeline, f)

if __name__ == '__main__':
    train(sys.argv[1])