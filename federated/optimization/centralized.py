import tensorflow as tf
from federated.data.data_preprocessing import get_datasets
from federated.models.models import (
    create_cnn_model,
    create_dense_model,
    create_softmax_model,
)
from federated.utils.training_loops import centralized_training_loop

MODELS = {
    "ann": create_dense_model,
    "cnn": create_cnn_model,
    "softmax_regression": create_softmax_model,
}

OPTIMIZERS = {
    "adam": lambda lr: tf.keras.optimizers.Adam(learning_rate=lr),
    "sgd": lambda lr: tf.keras.optimizers.SGD(learning_rate=lr),
}


def centralized_pipeline(
    name: str,
    output: str,
    epochs: int,
    batch_size: int,
    optimizer: str,
    model: str,
    learning_rate: float,
) -> None:
    """Function runs centralized training pipeline. Also logs traning configurations used during training.

    Args:
        name (str): Name of the experiment.\n
        output (str): Where to save config files. Defaults to history.\n
        epochs (int): Number of epochs. Defaults to 15.\n
        batch_size (int): Batch size. Defaults to 32.\n
        optimizer (str): Which optimizer to use. Defaults to sgd.\n
        model (str): Which model to use.\n
        learning_rate (float): Learning rate to be used. Defaults to 1.0.\n
    """

    model = MODELS[model]()
    optimizer = OPTIMIZERS[optimizer](learning_rate)

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
    )

    with open(f"{output}/logdir/{name}/training_config.csv", "w+") as f:
        f.writelines("name,training_time,epochs,time_per_epoch,optimizer\n")
        f.writelines(
            f"{name},{training_time},{epochs},{training_time/epochs},{optimizer}"
        )
        f.close()
