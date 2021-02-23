import tensorflow as tf
from federated.utils.training_loops import centralized_training_loop
from federated.data.mitbih_data_preprocessing import get_datasets
from federated.models.mitbih_model import (
    create_cnn_model,
    create_dense_model,
    create_new_cnn_model,
)


def centralized_pipeline(
    name,
    output,
    epochs,
    batch_size,
    optimizer,
    decay_epochs=None,
    learning_rate_decay=0,
):
    """
    Function runs centralized training pipeline
    """
    train_dataset, test_dataset = get_datasets(
        train_batch_size=batch_size, centralized=True
    )

    model = create_new_cnn_model()

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
    centralized_pipeline(name, "history", 15, 32, "sgd")