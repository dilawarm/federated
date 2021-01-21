import tensorflow as tf
from federated.utils.training_loops import centralized_training_loop
from federated.utils.mitbih_data_preprocessing import get_centralized_datasets
from federated.models.mitbih_model import create_cnn_model


def centralized_pipeline(name, output, epochs, batch_size, optimizer):
    """
    Function runs centralized training pipeline
    """
    train_dataset, test_dataset = get_centralized_datasets(
        train_batch_size=batch_size,
    )

    model = create_cnn_model()
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=optimizer,
        metrics=["accuracy"],
    )

    centralized_training_loop(
        model,
        train_dataset,
        name,
        epochs,
        output,
        save_model=True,
        validation_dataset=test_dataset,
    )


if __name__ == "__main__":
    centralized_pipeline(
        "test_v2",
        "history",
        5,
        32,
        tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
    )
