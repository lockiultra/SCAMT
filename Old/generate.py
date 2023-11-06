from dgllife.model import load_pretrained
import rdkit.Chem as Chem

def generate():
    model = load_pretrained('DGMG_ZINC_canonical')
    return model(rdkit_mol=True)

def get_valid_smiles():
    mol = None
    while mol is None:
        smiles = generate()
        mol = Chem.MolFromSmiles(smiles)
    return smiles

if __name__ == '__main__':
    smiles_generate = get_valid_smiles()
    print(smiles_generate)