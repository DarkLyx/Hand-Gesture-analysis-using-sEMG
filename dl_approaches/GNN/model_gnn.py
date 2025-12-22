import tensorflow as tf
from tensorflow.keras import layers, models, Input # type: ignore
import common.config as cfg

class LearnableGraphConv(layers.Layer):
    def __init__(self, output_dim, num_nodes, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.num_nodes = num_nodes

    def build(self, input_shape):
        self.A = self.add_weight(name='adj', shape=(self.num_nodes, self.num_nodes), initializer='glorot_uniform', trainable=True)
        self.bias = self.add_weight(name='bias', shape=(self.output_dim,), initializer='zeros', trainable=True)

    def call(self, inputs):
        return tf.nn.relu(tf.matmul(inputs, self.A) + self.bias)

def create_gnn_model(input_shape, num_classes):
    """Complete hybrid GNN (spatial) + CNN (temporal) architecture, compiled and ready for training."""
    inputs = Input(shape=input_shape)
    
    # 1. Graph Convolution (Spatial)
    # Learns spatial relationships between the sEMG sensors
    x = LearnableGraphConv(output_dim=input_shape[-1], num_nodes=input_shape[-1])(inputs)
    x = layers.BatchNormalization()(x)
    
    # 2. CNN Layers (Temporal)
    x = layers.Conv1D(64, 5, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(256, 3, padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.4)(x)
    
    # 3. Classification 
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # 4. Compilation 
    model = models.Model(inputs=inputs, outputs=outputs, name="GNN_sEMG")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model