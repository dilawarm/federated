from tensorflow.keras import models, layers

def create_cnn_model():
    """
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv1d (Conv1D)              (None, 182, 64)           448       
    _________________________________________________________________
    batch_normalization (BatchNo (None, 182, 64)           256       
    _________________________________________________________________
    max_pooling1d (MaxPooling1D) (None, 91, 64)            0         
    _________________________________________________________________
    conv1d_1 (Conv1D)            (None, 89, 64)            12352     
    _________________________________________________________________
    batch_normalization_1 (Batch (None, 89, 64)            256       
    _________________________________________________________________
    max_pooling1d_1 (MaxPooling1 (None, 45, 64)            0         
    _________________________________________________________________
    conv1d_2 (Conv1D)            (None, 43, 64)            12352     
    _________________________________________________________________
    batch_normalization_2 (Batch (None, 43, 64)            256       
    _________________________________________________________________
    max_pooling1d_2 (MaxPooling1 (None, 22, 64)            0         
    _________________________________________________________________
    flatten (Flatten)            (None, 1408)              0         
    _________________________________________________________________
    dense (Dense)                (None, 64)                90176     
    _________________________________________________________________
    dense_1 (Dense)              (None, 32)                2080      
    _________________________________________________________________
    dense_2 (Dense)              (None, 5)                 165       
    =================================================================
    Total params: 118,341
    Trainable params: 117,957
    Non-trainable params: 384
    _________________________________________________________________
    """
    model = models.Sequential([
        layers.Convolution1D(filters=64, kernel_size=6, activation="relu", input_shape=[187, 1]),
        layers.BatchNormalization(),
        layers.MaxPool1D(pool_size=3, strides=2, padding="same"),
        layers.Convolution1D(filters=64, kernel_size=3, activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPool1D(pool_size=2, strides=2, padding="same"),
        layers.Convolution1D(filters=64, kernel_size=3, activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPool1D(pool_size=2, strides=2, padding="same"),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(5, activation="softmax")
    ])

    return model

if __name__ == "__main__":
    model = create_cnn_model()
    print(model.summary())