import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class EdgeNetwork(layers.Layer):
    def build(self, input_shape: tf.TensorShape):
        """
        Builds the model by initializing the atom_dim, bond_dim, kernel, and bias weights.
        
        Args:
            input_shape (tf.TensorShape): The shape of the input tensor.
        
        Returns:
            None
        """
        self.atom_dim = input_shape[0][-1]
        self.bond_dim = input_shape[1][-1]
        self.kernel  = self.add_weight(
            shape=(self.bond_dim, self.atom_dim * self.atom_dim),
            initializer="glorot_uniform",
            name="kernel",
        )
        self.bias = self.add_weight(
            shape=(self.atom_dim * self.atom_dim),
            initializer="zeros",
            name="bias",
        )
        self.built = True

    def call(self, inputs: tuple) -> tf.Tensor:
        """
        Compute the aggregated features for each atom based on the atom features, bond features, and pair indices.

        Args:
            inputs (tuple): A tuple containing the atom features, bond features, and pair indices.
                - atom_features (tf.Tensor): A tensor representing the atom features.
                - bond_features (tf.Tensor): A tensor representing the bond features.
                - pair_indices (tf.Tensor): A tensor representing the pair indices.

        Returns:
            tf.Tensor: A tensor representing the aggregated features for each atom.
        """
        atom_features, bond_features, pair_indices = inputs
        bond_features = tf.matmul(bond_features, self.kernel) + self.bias
        bond_features = tf.reshape(bond_features, (-1, self.atom_dim, self.atom_dim))
        atom_features_neighbors = tf.gather(atom_features, pair_indices[:, 1])
        atom_features_neighbors = tf.expand_dims(atom_features_neighbors, axis=-1)
        transformed_features = tf.matmul(bond_features, atom_features_neighbors)
        transformed_features = tf.squeeze(transformed_features, axis=-1)
        aggregated_features = tf.math.unsorted_segment_sum(
            transformed_features,
            pair_indices[:, 0],
            num_segments=tf.shape(atom_features)[0]
        )
        return aggregated_features

class MessagePassing(layers.Layer):
    def __init__(self, units: int, steps: int = 4, **kwargs):
        """
        Initializes a new instance of the class.

        Args:
            units (int): The number of units.
            steps (int, optional): The number of steps. Defaults to 4.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        super().__init__(**kwargs)
        self.units = units
        self.steps = steps

    def build(self, input_shape: tf.TensorShape):
        """
        Builds the model by setting the atom dimension, message step, pad length, and update step.

        Parameters:
            input_shape (tf.TensorShape): The shape of the input tensor.

        Returns:
            None
        """
        self.atom_dim = input_shape[0][-1]
        self.message_step = EdgeNetwork()
        self.pad_length = max(0, self.units - self.atom_dim)
        self.update_step = layers.GRUCell(self.atom_dim + self.pad_length)
        self.built = True

    def call(self, inputs: tuple) -> tf.Tensor:
        """
        Applies a series of message and update steps to the input atom and bond features.

        Args:
            inputs (tuple): A tuple containing the atom features, bond features, and pair indices.

        Returns:
            tf.Tensor: The updated atom features after applying the message and update steps.
        """
        atom_features, bond_features, pair_indices = inputs
        atom_features_updated = tf.pad(atom_features, [[0, 0], [self.pad_length, 0]])
        for i in range(self.steps):
            atom_features_aggregated = self.message_step([atom_features_updated, bond_features, pair_indices])
            atom_features_updated, _ = self.update_step(atom_features_aggregated, atom_features_updated)
        return atom_features_updated

class PartitionPadding(layers.Layer):
    def __init__(self, batch_size: int, **kwargs):
        """
        Initializes a new instance of the class.

        Args:
            batch_size (int): The size of the batch.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size

    def call(self, inputs: tuple) -> tf.Tensor:
        """
        Given a tuple of inputs, this function partitions the atom features based on the molecule indicator, 
        stacks the partitioned atom features, and returns the gathered atom features.

        Parameters:
        - inputs: A tuple containing atom features and molecule indicator.

        Returns:
        - A Tensor containing the gathered atom features.
        """
        atom_features, molecule_indicator = inputs
        atom_features_partitioned = tf.dynamic_partition(atom_features, molecule_indicator, self.batch_size)
        num_atoms = [tf.shape(x)[0] for x in atom_features_partitioned]
        max_num_atoms = tf.reduce_max(num_atoms)
        atom_features_stacked = tf.stack(
            [
                tf.pad(f, [(0, max_num_atoms - n), (0, 0)]) for f, n in zip(atom_features_partitioned, num_atoms)
            ],
            axis=0
        )
        gather_indices = tf.where(tf.reduce_sum(atom_features_stacked, (1, 2)) != 0)
        gather_indices = tf.squeeze(gather_indices, axis=-1)
        return tf.gather(atom_features_stacked, gather_indices, axis=0)

class TransformerEncoderReadout(layers.Layer):
    def __init__(self, num_heads: int = 8, embed_dim: int = 64, dense_dim: int = 512, batch_size: int = 32, **kwargs):
        """
        Initializes an instance of the class with the specified number of heads, embedding dimension, dense dimension, and batch size.

        Args:
            num_heads (int, optional): The number of attention heads. Defaults to 8.
            embed_dim (int, optional): The embedding dimension. Defaults to 64.
            dense_dim (int, optional): The dense dimension. Defaults to 512.
            batch_size (int, optional): The batch size. Defaults to 32.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        super().__init__(**kwargs)
        self.partition_padding = PartitionPadding(batch_size)
        self.attention = layers.MultiHeadAttention(num_heads, embed_dim)
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(dense_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.average_pooling = layers.GlobalAveragePooling1D()

    def call(self, inputs: tuple) -> tf.Tensor:
        """
        Calls the model on the given inputs.

        Args:
            inputs (tuple): The inputs to the model.

        Returns:
            tf.Tensor: The output of the model.
        """
        x = self.partition_padding(inputs)
        padding_mask = tf.reduce_any(tf.not_equal(x, 0.0), axis=-1)
        padding_mask = padding_mask[:, tf.newaxis, tf.newaxis, :]
        attention_output = self.attention(x, x, attention_mask=padding_mask)
        proj_input = self.layernorm_1(x + attention_output)
        proj_output = self.layernorm_2(proj_input + self.dense_proj(proj_input))
        return self.average_pooling(proj_output)

def MPNNModel(atom_dim: int, bond_dim: int, batch_size: int = 128, message_units: int = 64, message_steps: int = 4, num_attention_heads: int = 8, dense_units: int = 512) -> keras.Model:
    """
    Creates a message passing neural network (MPNN) model for molecule property prediction.

    Args:
        atom_dim (int): The dimensionality of the atom features.
        bond_dim (int): The dimensionality of the bond features.
        batch_size (int, optional): The batch size. Defaults to 128.
        message_units (int, optional): The number of units in the message passing layer. Defaults to 64.
        message_steps (int, optional): The number of message passing steps. Defaults to 4.
        num_attention_heads (int, optional): The number of attention heads in the transformer encoder readout layer. Defaults to 8.
        dense_units (int, optional): The number of units in the dense layer. Defaults to 512.

    Returns:
        keras.Model: The compiled MPNN model.
    """
    atom_features = layers.Input((atom_dim,), dtype=tf.float32, name="atom_features")
    bond_features = layers.Input((bond_dim,), dtype=tf.float32, name="bond_features")
    pair_indices = layers.Input((2,), dtype=tf.int32, name="pair_indices")
    molecule_indicator = layers.Input((), dtype=tf.int32, name="molecule_indicator")
    x = MessagePassing(message_units, message_steps)([atom_features, bond_features, pair_indices])
    x = TransformerEncoderReadout(num_attention_heads, message_units, dense_units, batch_size)([x, molecule_indicator])
    x = layers.Dense(dense_units, activation='elu')(x)
    x = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=[atom_features, bond_features, pair_indices, molecule_indicator], outputs=[x])
    return model