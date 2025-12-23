import tensorflow as tf
from tensorflow.keras import layers, models, Input # type: ignore
import common.config as cfg

def create_vit_model(input_shape, num_classes):
    """Architecture Vision Transformer (ViT) complète et compilée."""
    inputs = Input(shape=input_shape)
    
    # 1. Patching & Embedding
    patches = layers.Conv1D(filters=cfg.PROJECTION_DIM, kernel_size=cfg.PATCH_SIZE, strides=cfg.PATCH_SIZE, padding='valid')(inputs)
    
    positions = tf.range(start=0, limit=patches.shape[1], delta=1)
    position_embedding = layers.Embedding(input_dim=patches.shape[1], output_dim=cfg.PROJECTION_DIM)(positions)
    x = patches + position_embedding

    # 2. Transformer Encoder Blocks 
    for _ in range(cfg.TRANSFORMER_LAYERS):
        # Attention Layer
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        attention_output = layers.MultiHeadAttention(num_heads=cfg.NUM_HEADS, key_dim=cfg.PROJECTION_DIM, dropout=0.1)(x1, x1)
        x2 = layers.Add()([attention_output, x])
        
        # Feed Forward Network (MLP)
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = layers.Dense(cfg.PROJECTION_DIM * 2, activation=tf.nn.gelu)(x3)
        x3 = layers.Dropout(0.1)(x3)
        x3 = layers.Dense(cfg.PROJECTION_DIM, activation=tf.nn.gelu)(x3)
        x3 = layers.Dropout(0.1)(x3)
        
        # Skip Connection finale
        x = layers.Add()([x3, x2])

    # 3. Classification Head 
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(cfg.MLP_HEAD_UNITS[0], activation=tf.nn.gelu)(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    # 4. Compilation 
    model = models.Model(inputs=inputs, outputs=outputs, name="ViT_sEMG")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model