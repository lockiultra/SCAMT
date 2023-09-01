from dgllife.model import load_pretrained

def generate():
    model = load_pretrained('DGMG_ZINC_canonical')
    return model(rdkit_mol=True)

if __name__ == '__main__':
    smiles_generate = generate()
    print(smiles_generate)