import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from rdkit import Chem

from Featurizers import AtomFeaturizer, BondFeaturizer
from MPNN import MPNNModel

class DiseasePipeline:
  def __init__(self: 'DiseasePipeline'):
    """
    Initializes the class instance.
    """
    self.diseases = ['cardiovascular_disease', 'digestive_system_disease', 'immune_system_disease', 'mental_and_behavioural_disorder', 'metabolic_disease', 'nervous_system_disease', 'skin_and_connective_tissue_disease', 'urinary_system_disease']
    self.atom_featurizer = AtomFeaturizer(
      allowable_sets={
        "symbol": {'Al','As','Au','B','Br','C','Cl','Co','F','Fe','Gd','H','I','K','Mn','Mo','N','Na','O','P','Pd','Pt','S','Se','Si','Zn'},
        "n_valence": {0, 1, 2, 3, 4, 5, 6},
        "n_hydrogens": {0, 1, 2, 3, 4},
        "hybridization": {'s', 'sp', 'sp2', 'sp3', 'sp3d', 'sp3d2'},
      }
    )
    self.bond_featurizer = BondFeaturizer(
      allowable_sets={
        "bond_type": {'single', 'double', 'triple', 'aromatic'},
        "conjugated": {False, True},
      }
    )
    self.curr_df = None
    self.models = {disease: MPNNModel(atom_dim=44, bond_dim=6) for disease in self.diseases}
    self.is_trained = False
    self.train_history = dict()

  def train(self, data: pd.DataFrame) -> None:
    """
    Trains the model using the provided data.

    Args:
        data (pd.DataFrame): The input data for training.

    Returns:
        None
    """
    self.data = data.copy()
    for i, disease in enumerate(self.diseases):
      print(f'\n({i}) ======={disease}=======\n')
      self.curr_df = self.data.copy()
      self.curr_df['Disease'] = self.curr_df['Disease'].replace({x: 1 if x == disease else 0 for x in self.diseases})
      self.curr_df = self.curr_df.drop_duplicates(subset=['Drug'])
      x_train, y_train, x_val, y_val = self.__get_train_val_data(self.curr_df)
      train_dataset = self.__get_dataset(x_train, y_train, disease)
      val_dataset = self.__get_dataset(x_val, y_val, disease)
      self.models[disease].compile(
        loss=keras.losses.BinaryCrossentropy(),
        optimizer=keras.optimizers.AdamW(learning_rate=3e-4),
        metrics=[keras.metrics.AUC(name='AUC')],
      )
      history = self.models[disease].fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=40,
        verbose=2,
      )
      self.train_history[disease] = history
    self.is_trained = True

  def predict(self, smiles: str) -> dict:
    """
    Predicts the disease for a given SMILES string.

    Args:
        smiles (str): The SMILES string to predict the disease for.

    Returns:
        dict: A dictionary containing the predicted disease for each disease model.
    """
    if not self.is_trained:
      print('Error! Model is not trained')
      return
    result = {disease: None for disease in self.diseases}
    g = self.graph_from_smiles(smiles)
    dataset = tf.data.Dataset.from_tensors(((g), (1))).map(self.prepare_batch, -1).prefetch(-1)
    for disease, model in zip(self.models.keys(), self.models.values()):
      result[disease] = model.predict(dataset)
    return result


  def __get_train_val_data(self, data: pd.DataFrame) -> tuple:
    """
    Generate the training and validation data from the given DataFrame.

    Parameters:
    - data (pd.DataFrame): The input DataFrame containing the data.

    Returns:
    - tuple: A tuple containing the training and validation data.
      - x_train (np.ndarray): The training data represented as graphs.
      - y_train (pd.Series): The corresponding labels for the training data.
      - x_val (np.ndarray): The validation data represented as graphs.
      - y_val (pd.Series): The corresponding labels for the validation data.
    """
    data = data.dropna()
    permutation = np.random.permutation(np.arange(data.shape[0]))
    train_index = permutation#[:int(data.shape[0] * 0.8)]
    x_train = self.graph_from_smiles(data.iloc[train_index].Smiles)
    y_train = data.iloc[train_index].Disease
    val_index = permutation[int(data.shape[0] * 0.8):]
    x_val = self.graph_from_smiles(data.iloc[val_index].Smiles)
    y_val = data.iloc[val_index].Disease
    return (x_train, y_train, x_val, y_val)

  def __get_dataset(self, X, y, batch_size: int = 128, shuffle: bool = False) -> tf.data.Dataset:
    """
    Generates a TensorFlow dataset from the given input data.

    Args:
        X: The input data.
        y: The target labels.
        batch_size: The batch size for the dataset. Defaults to 128.
        shuffle: Whether to shuffle the dataset. Defaults to False.

    Returns:
        A TensorFlow dataset object.
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, (y)))
    if shuffle:
      dataset = dataset.shuffle(1024)
    return dataset.batch(batch_size).map(self.prepare_batch, -1).prefetch(-1)

  def prepare_batch(self, x_batch: tuple, y_batch: np.ndarray) -> tuple:
    """
    Prepares a batch of data for training or evaluation.

    Args:
        x_batch (tuple): A tuple containing atom_features, bond_features, and pair_indices.
        y_batch (np.ndarray): An array containing the target values.

    Returns:
        tuple: A tuple containing the prepared batch data and the target values.
    """
    atom_features, bond_features, pair_indices = x_batch
    num_atoms = atom_features.row_lengths()
    num_bonds = bond_features.row_lengths()
    molecule_indices = tf.range(len(num_atoms))
    molecule_indicator = tf.repeat(molecule_indices, num_atoms)
    gather_indices = tf.repeat(molecule_indices[:-1], num_bonds[1:])
    increment = tf.cumsum(num_atoms[:-1])
    increment = tf.pad(tf.gather(increment, gather_indices), [(num_bonds[0], 0)])
    pair_indices = pair_indices.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    pair_indices = pair_indices + increment[:, tf.newaxis]
    atom_features = atom_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    bond_features = bond_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    return (atom_features, bond_features, pair_indices, molecule_indicator), y_batch

  def molecule_from_smiles(self, smiles: str) -> Chem.Mol:
    """
    Generate a molecule object from a SMILES string.

    Args:
        smiles (str): The SMILES string representation of the molecule.

    Returns:
        Chem.Mol: The molecule object generated from the SMILES string.
    """
    molecule = Chem.MolFromSmiles(smiles, sanitize=False)
    flag = Chem.SanitizeMol(molecule, catchErrors=True)
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
        Chem.SanitizeMol(molecule, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ flag)
    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    return molecule

  def graph_from_molecule(self, molecule: Chem.Mol) -> tuple:
      """
      Generate a graph representation from a given molecule.

      Args:
          molecule (Chem.Mol): The molecule to generate the graph from.

      Returns:
          tuple: A tuple containing the atom features, bond features, and pair indices.
              - atom_features (np.array): An array of atom features.
              - bond_features (np.array): An array of bond features.
              - pair_indices (np.array): An array of pair indices.
      """
      atom_features = []
      bond_features = []
      pair_indices = []
      for atom in molecule.GetAtoms():
          atom_features.append(self.atom_featurizer.encode(atom))
          pair_indices.append((atom.GetIdx(), atom.GetIdx()))
          bond_features.append(self.bond_featurizer.encode(None))
          for neighbor in atom.GetNeighbors():
              bond = molecule.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
              pair_indices.append((atom.GetIdx(), neighbor.GetIdx()))
              bond_features.append(self.bond_featurizer.encode(bond))
      return np.array(atom_features), np.array(bond_features), np.array(pair_indices)

  def graph_from_smiles(self, smiles_list: list) -> tuple:
      """
      Generate a graph from a list of SMILES strings.

      Args:
          smiles_list (list): A list of SMILES strings representing molecules.

      Returns:
          tuple: A tuple containing three TensorFlow ragged tensors:
                 - atom_features (tf.Tensor): A ragged tensor of atom features.
                 - bond_features (tf.Tensor): A ragged tensor of bond features.
                 - pair_indices (tf.Tensor): A ragged tensor of pair indices.
      """
      atom_features_list = []
      bond_features_list = []
      pair_indices_list = []
      for smiles in smiles_list:
          molecule = self.molecule_from_smiles(smiles)
          atom_features, bond_features, pair_indices = self.graph_from_molecule(molecule)
          atom_features_list.append(atom_features)
          bond_features_list.append(bond_features)
          pair_indices_list.append(pair_indices)
      return (
          tf.ragged.constant(atom_features_list, dtype=tf.float32),
          tf.ragged.constant(bond_features_list, dtype=tf.float32),
          tf.ragged.constant(pair_indices_list, dtype=tf.int64),
      )