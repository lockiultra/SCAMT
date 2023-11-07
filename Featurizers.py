import numpy as np
import rdkit.Chem as Chem

class Featurizer:
    def __init__(self, allowable_sets: dict):
        """
        Initializes the object with the given allowable sets.

        Parameters:
            allowable_sets (dict): A dictionary containing the allowable sets for each key.

        Returns:
            None
        """
        self.dim = 0
        self.feature_mapping = {}
        for k, s in allowable_sets.items():
            s = sorted(list(s))
            self.feature_mapping[k] = dict(zip(s, range(self.dim, self.dim + len(s))))
            self.dim += len(s)

    def encode(self, inputs: any) -> np.ndarray:
        """
        Encodes the given inputs using the feature mappings defined in the object.

        Parameters:
            inputs (any): The inputs to be encoded.

        Returns:
            numpy.ndarray: The encoded output.
        """
        output = np.zeros((self.dim,))
        for name_feature, feature_mapping in self.feature_mapping.items():
            feature = getattr(self, name_feature)(inputs)
            if feature not in feature_mapping:
                continue
            output[feature_mapping[feature]] = 1.0
        return output


class AtomFeaturizer(Featurizer):
    def __init__(self, allowable_sets: list):
        """
        Initializes the class instance with the given allowable sets.

        :param allowable_sets: A list of allowable sets.
        """
        super().__init__(allowable_sets)

    def symbol(self, atom: Chem.Atom) -> str:
        """
        Returns the symbol of the given atom.

        Parameters:
            atom (Atom): The atom object for which to retrieve the symbol.

        Returns:
            str: The symbol of the given atom.
        """
        return atom.GetSymbol()

    def n_valence(self, atom: Chem.Atom) -> int:
        """
        Calculate the total valence of an atom.

        Args:
            atom (Atom): The atom for which to calculate the total valence.

        Returns:
            int: The total valence of the atom.
        """
        return atom.GetTotalValence()

    def n_hydrogens(self, atom: Chem.Atom) -> int:
        """
        Get the total number of hydrogen atoms attached to the given atom.

        Parameters:
            atom (Chem.Atom): The atom for which to get the total number of hydrogen atoms.

        Returns:
            int: The total number of hydrogen atoms attached to the atom.
        """
        return atom.GetTotalNumHs()

    def hybridization(self, atom: Chem.Atom) -> str:
        """
        Returns the hybridization of an atom.

        Parameters:
            atom (Atom): The atom object for which to get the hybridization.

        Returns:
            str: The hybridization of the atom as a lowercase string.
        """
        return atom.GetHybridization().name.lower()


class BondFeaturizer(Featurizer):
    def __init__(self, allowable_sets: list):
        """
        Initializes the class instance with the given allowable sets.

        :param allowable_sets: A list of allowable sets.
        """
        super().__init__(allowable_sets)

    def encode(self, bond: Chem.Bond) -> np.ndarray:
        """
        Encode the given bond into a vector representation.

        Parameters:
            bond (Bond): The bond to be encoded.

        Returns:
            numpy.ndarray: The encoded vector representation of the bond.
        """
        output = np.zeros((self.dim,))
        if bond is None:
            output[-1] = 1.0
            return output
        output = super().encode(bond)
        return output

    def bond_type(self, bond: Chem.Bond) -> str:
        """
        Returns the bond type of a given bond.

        Parameters:
            bond (object): The bond object for which to retrieve the bond type.

        Returns:
            str: The lowercase string representation of the bond type.
        """
        return bond.GetBondType().name.lower()

    def conjugated(self, bond: Chem.Bond) -> bool:
        """
        Returns whether the given bond is conjugated.

        Parameters:
            bond (Bond): The bond to check for conjugation.

        Returns:
            bool: True if the bond is conjugated, False otherwise.
        """
        return bond.GetIsConjugated()