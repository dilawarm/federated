import tensorflow as tf
from federated.data.data_preprocessing import get_datasets
from federated.models.models import (
    create_cnn_model,
    create_dense_model,
    create_new_cnn_model,
)
from federated.utils.training_loops import centralized_training_loop


def centralized_pipeline(
    name: str,
    output: str,
    epochs: int,
    batch_size: int,
    optimizer: tf.keras.optimizers.Optimizer,
    model: tf.keras.Model,
    decay_epochs: int = None,
    learning_rate_decay: float = 0,
) -> None:
    """Function runs centralized training pipeline. Also logs traning configurations used during training.

    Args:
        name (str): Name of the experiment.\n
        output (str): Where to save config files.\n
        epochs (int): Number of epochs.\n
        batch_size (int): Batch size.\n
        optimizer (tf.keras.optimizers.Optimizer): Which optimizer to use.\n
        model (tf.keras.Model): Which model to use.\n
        decay_epochs (int, optional): Frequency of learning rate decay. Defaults to None.\n
        learning_rate_decay (float, optional): Rate of learning rate decay. Defaults to 0.\n
    """
    train_dataset, test_dataset, _ = get_datasets(
        train_batch_size=batch_size, centralized=True
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )

    _, training_time = centralized_training_loop(
        model,
        train_dataset,
        name,
        epochs,
        output,
        save_model=True,
        validation_dataset=test_dataset,
        decay_epochs=decay_epochs,
        learning_rate_decay=learning_rate_decay,
    )

    with open(f"history/logdir/{name}/training_config.csv", "w+") as f:
        f.writelines("name,training_time,epochs,time_per_epoch,optimizer\n")
        f.writelines(
            f"{name},{training_time},{epochs},{training_time/epochs},{optimizer}"
        )
        f.close()


if __name__ == "__main__":
    name = input("Experiment name: ")
    centralized_pipeline(name, "history", 1, 32, "sgd", create_new_cnn_model())
