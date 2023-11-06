import pickle
from DiseasePipeline import DiseasePipeline

def predict(smiles):
    dp = DiseasePipeline()
    with open('./Models/disease_pipeline', 'rb') as f:
        dp = pickle.load(f)
    return dp.predict(smiles)

if __name__ == '__main__':
    print(predict(sys.argv[1]))