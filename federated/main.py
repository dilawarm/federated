from federated.optimization.centralized import centralized_pipeline
from federated.optimization.federated import federated_pipeline

from typing import List
from itertools import chain, repeat
import argparse


def check_type(x, inp_type):
    try:
        inp_type(x)
        return True
    except:
        return False


def validate_type_input(input_string: str, default, inp_type):
    prompts = chain(
        [input_string + f"{default}: "],
        repeat(f"Input must be of type {inp_type}. Try again: "),
    )
    replies = map(input, prompts)
    valid_response = next(filter(lambda x: check_type(x, inp_type) or x == "", replies))

    if valid_response == "":
        return default
    return inp_type(valid_response)


def validate_options_input(input_string: str, default: str, options: List[str]):
    prompts = chain(
        [input_string + f"{options}: "],
        repeat(f"Input must be one of {options}. Try again: "),
    )
    replies = map(input, prompts)
    valid_response = next(filter(lambda x: (x in options + [""]), replies))

    if valid_response == "":
        return default

    return valid_response


def main():
    parser = argparse.ArgumentParser(
        description="Experimentation pipeline for federated."
    )

    parser.add_argument(
        "-l",
        "--learning_approach",
        choices=["centralized", "federated"],
        metavar="",
        required=True,
        help="Learning apporach (centralized, federated).",
    )
    parser.add_argument(
        "-n",
        "--experiment_name",
        type=str,
        metavar="",
        required=True,
        help="The name of the experiment.",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=15,
        metavar="",
        help="Number of global epochs.",
    )
    parser.add_argument(
        "-op",
        "--optimizer",
        choices=["adam", "sgd"],
        metavar="",
        default="sgd",
        help="Server optimizer (adam, sgd).",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        metavar="",
        default=32,
        help="The batch size.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        metavar="",
        default="history",
        help="The name of the output folder for where the experiment is saved.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=["ann", "softmax_regression", "cnn"],
        metavar="",
        required=True,
        help="The model to be trained with the learning approach (ann, softmax_regression, cnn).",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        metavar="",
        default=1.0,
        help="Learning rate for server optimizer.",
    )

    args = parser.parse_args()

    if args.learning_approach == "centralized":
        centralized_pipeline(
            name=args.experiment_name,
            output=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            optimizer=args.optimizer,
            model=args.model,
            learning_rate=args.learning_rate,
        )
    else:
        print("Press ENTER to skip input and use default values.", end="\n\n")

        client_epochs = None
        client_optimizer = None
        client_lr = None
        iterations = None
        v = None
        noise_multiplier = None
        clipping_norm = None

        aggregation_method = validate_options_input(
            f"Aggregation Method. OPTIONS: ",
            "fedavg",
            ["fedsgd", "fedavg", "rfa"],
        )

        client_weighting = validate_options_input(
            f"Client weighting. OPTIONS: ", "NUM_EXAMPLES", ["NUM_EXAMPLES", "UNIFORM"]
        )

        if aggregation_method in ["fedavg", "rfa"]:

            client_epochs = validate_type_input(
                "Enter number of client epochs. DEFAULT : ", 10, int
            )

            client_optimizer = validate_options_input(
                f"Client optimizer. OPTIONS: ",
                "sgd",
                ["adam", "sgd"],
            )

            client_lr = validate_type_input(
                "Learning rate for client optimizer. DEFAULT: ", 0.02, float
            )

            if aggregation_method == "rfa":
                iterations = validate_type_input(
                    "Number of calls to the Secure Average Oracle. DEFAULT: ", 3, int
                )
                v = validate_type_input("L2 Threshold. DEFAULT: ", 1e-6, float)

        if aggregation_method != "rfa":
            dp = validate_options_input(
                f"Differential Privacy. OPTIONS: ", "False", ["True", "False"]
            )

            if eval(dp):
                noise_multiplier = validate_type_input(
                    "Noise multiplier. DEFAULT: ",
                    0.5,
                    float,
                )
                clipping_norm = validate_type_input(
                    "Clipping norm. DEFAULT: ",
                    0.75,
                    float,
                )

                client_weighting = "UNIFORM"
                print(
                    "Client weighting is set to be UNIFORM because you chose to train with differential privacy."
                )

        data_dist = validate_options_input(
            "Enter data distribution. OPTIONS: ",
            "non_iid",
            ["non_iid", "uniform", "class_distributed"],
        )

        if data_dist == "class_distributed":
            number_of_clients = 5
            print("Number of clients is set as 5 because you chose class_distributed.")
        else:
            number_of_clients = validate_type_input(
                "Number of clients. DEFAULT: ", 10, int
            )

        number_of_clients_per_round = validate_type_input(
            "Number of participating clients per round. DEFAULT: ", 5, int
        )

        if number_of_clients_per_round > number_of_clients:
            raise ValueError(
                "Number of participating clients can't be larger than the number of clients. Try again."
            )

        compression = validate_options_input(
            f"Compression. OPTIONS: ", "False", ["True", "False"]
        )

        seed = validate_type_input(f"Random seed. Default: ", None, int)

        federated_pipeline(
            name=args.experiment_name,
            aggregation_method=aggregation_method,
            client_weighting=client_weighting,
            keras_model_fn=args.model,
            server_optimizer_fn=args.optimizer,
            server_optimizer_lr=args.learning_rate,
            client_optimizer_fn=client_optimizer,
            client_optimizer_lr=client_lr,
            data_selector=data_dist,
            output=args.output,
            client_epochs=client_epochs,
            batch_size=args.batch_size,
            number_of_clients=number_of_clients,
            number_of_clients_per_round=number_of_clients_per_round,
            number_of_rounds=args.epochs,
            iterations=iterations,
            v=v,
            compression=compression,
            dp=dp,
            noise_multiplier=noise_multiplier,
            clipping_value=clipping_norm,
            seed=seed,
        )


if __name__ == "__main__":
    main()