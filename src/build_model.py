import tensorflow as tf
from keras import layers, models
from tokenizer import VOCAB

def protein_cnn_model(
        max_len: int=512,
        embedding_dim: int = 128,
        num_classes: int=10,
        conv_filters: list = [64, 128, 256, 512],
        kernel_size: int = 3,
        dropout_rate: float = 0.3,
):
    """
    Build a DeepSEA-like CNN model for protein-sequence classification.

    Parameters:
    - max_len: Maximum length of input sequences.
    - embedding_dim: Dimension of the embedding layer.
    - num_classes: Number of output classes.
    - conv_filters: List of filter sizes for convolutional layers.
    - kernel_size: Size of the convolutional kernels.
    - dropout_rate: Dropout rate for regularization.

    Returns:
    - A compiled Keras model.
    """
    inputs = layers.Input(shape=(max_len,), dtype='int32')
    
    # Embedding layer
    x = layers.Embedding(input_dim=len(VOCAB), output_dim=embedding_dim,mask_zero=True, input_length=max_len)(inputs)
    
    # Convolutional layers(feature extraction)
    for filters in conv_filters:
        x = layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(rate=dropout_rate)(x)
    
    # Flatten and Dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(rate=dropout_rate)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='protein_cnn_model')
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_cross_entropy', metrics=['accuracy'])
    
    return model

if __name__ == "__main__":
    model = protein_cnn_model()
    model.summary()