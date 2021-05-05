import argparse
import json
import time
from itertools import chain, repeat
from operator import attrgetter
from time import sleep
from typing import Any, List

import emoji
from alive_progress import alive_bar, config_handler

from federated.optimization.centralized import centralized_pipeline
from federated.optimization.federated import federated_pipeline

boldify = lambda s: "\033[1m" + str(s) + "\033[0m"
boldify.__doc__ = """Function for producing boldface strings."""


def remove_slash(path: str) -> str:
    """Remove slash in the end of path if present.

    Args:
        path (str): Path.

    Returns:
        str: Path without slash.
    """
    if path[-1] == "/":
        path = path[:-1]
    return path


def print_training_config(args: dict) -> None:
    """Function for printing out training configuration.

    Args:
        args (dict): Training configuration dictionary.
    """
    print(emoji.emojize("\nTraining Configuration :page_with_curl:", use_aliases=True))
    print(json.dumps(args, indent=4, sort_keys=True), end="\n\n")
    time.sleep(3)


def check_type(x: str, inp_type: type) -> bool:
    """Function for checking if a string can be converted to a certain type.

    Args:
        x (str): String to convert.\n
        inp_type (type) : Type.

    Returns:
        bool: If x can be converted or not.
    """
    try:
        inp_type(x)
        return True
    except:
        return False


def validate_type_input(input_string: str, default: Any, inp_type: type) -> Any:
    """Custom input with validation for ints, floats, and bools.

    Args:
        input_string (str): Prompt string for input.\n
        default (Any): Default value.\n
        inp_type (type): Type to convert to.

    Returns:
        Any: Converted input value.
    """

    prompts = chain(
        [input_string + f"(default: {boldify(default)}): "],
        repeat(f"Input must be of type {inp_type}. Try again: "),
    )
    replies = map(input, prompts)
    valid_response = next(filter(lambda x: check_type(x, inp_type) or x == "", replies))

    if valid_response == "":
        if inp_type == bool:
            return eval(default)
        return default

    if inp_type == bool:
        return eval(valid_response)

    return inp_type(valid_response)


def validate_options_input(input_string: str, default: str, options: List[str]) -> str:
    """Custom input with validation where we have options.

    Args:
        input_string (str): Prompt string for input.\n
        default (str): Default value.\n
        options (List[str]): Options for input.\n

    Returns:
        str: Input value.
    """
    prompts = chain(
        [input_string + f"{options}. (default: {boldify(default)}): "],
        repeat(f"Input must be one of {options}. Try again: "),
    )
    replies = map(input, prompts)
    valid_response = next(filter(lambda x: (x in options + [""]), replies))

    if valid_response == "":
        return default

    return valid_response


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Custom implementation of argparse.ArgumentDefaultsHelpFormatter."""

    def add_arguments(self, actions: dict) -> None:
        """Custom add_arguments of argparse.ArgumentDefaultsHelpFormatter.

        Args:
            actions (dict): Actions.
        """
        actions = sorted(actions, key=attrgetter("option_strings"))
        super(CustomFormatter, self).add_arguments(actions)


def main():
    """Main method for this project. Runs pipelines with user inputs, and writes configuration to json file.

    Raises:
        ValueError: If participating clients is larger than total number of clients.
    """
    parser = argparse.ArgumentParser(
        description=emoji.emojize(
            "Experimentation pipeline for federated :rocket:", use_aliases=True
        ),
        formatter_class=CustomFormatter,
    )

    for g in parser._action_groups:
        g._group_actions.sort(key=lambda x: x.dest)

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
        help="Path to the output folder where the experiment is going to be saved.",
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

    args.output = remove_slash(args.output)

    args_dict = vars(args)
    output = args_dict["output"]
    args.output = f"{output}/logdir/{args.experiment_name}"

    print(
        f"\n{emoji.emojize('Initializing pipeline... :chicken: :arrow_right: :hatching_chick: :arrow_right: :hatched_chick:', use_aliases=True)}"
    )
    config_handler.set_global(length=30, spinner="fish_bouncing")
    with alive_bar(200, length=40) as bar:
        for i in range(200):
            sleep(0.02)
            bar()

    if args.learning_approach == "centralized":

        print_training_config(args_dict)
        args.output = output

        training_time = centralized_pipeline(
            name=args.experiment_name,
            output=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            optimizer=args.optimizer,
            model=args.model,
            learning_rate=args.learning_rate,
        )
    else:
        print(
            emoji.emojize(
                "\nPress ENTER to skip input and use default values :fast_forward:",
                use_aliases=True,
            ),
            end="\n\n",
        )

        args_dict["aggregation_method"] = validate_options_input(
            f"Aggregation method. OPTIONS: ",
            "fedavg",
            ["fedsgd", "fedavg", "rfa"],
        )

        args_dict["client_weighting"] = validate_options_input(
            f"Client weighting. OPTIONS: ",
            "NUM_EXAMPLES",
            ["NUM_EXAMPLES", "UNIFORM"],
        )

        if args_dict["aggregation_method"] in ["fedavg", "rfa"]:

            args_dict["client_epochs"] = validate_type_input(
                "Number of client epochs. ", 10, int
            )

            args_dict["client_optimizer"] = validate_options_input(
                f"Client optimizer. OPTIONS: ",
                "sgd",
                ["adam", "sgd"],
            )

            args_dict["client_lr"] = validate_type_input(
                "Learning rate for client optimizer. ", 0.02, float
            )

            if args_dict["aggregation_method"] == "rfa":
                args_dict["rfa_iterations"] = validate_type_input(
                    "Number of calls to the Secure Average Oracle. ", 3, int
                )
                args_dict["l2_threshold"] = validate_type_input(
                    "L2 Threshold. ", 1e-6, float
                )

        if args_dict["aggregation_method"] != "rfa":
            args_dict["differentially_privacy"] = validate_type_input(
                f"Differential privacy. ", "False", bool
            )

            if args_dict["differentially_privacy"]:
                args_dict["noise_multiplier"] = validate_type_input(
                    "Noise multiplier. ",
                    0.5,
                    float,
                )
                args_dict["clipping_norm"] = validate_type_input(
                    "Clipping norm. ",
                    0.75,
                    float,
                )

                args_dict["client_weighting"] = "UNIFORM"
                print(
                    emoji.emojize(
                        "\n:exclamation: Client weighting is set to be UNIFORM because you chose to train with differential privacy.",
                        use_aliases=True,
                    ),
                    end="\n\n",
                )

        args_dict["compression"] = validate_type_input(
            f"Compress model. ", "False", bool
        )

        args_dict["data_dist"] = validate_options_input(
            "Data distribution. OPTIONS: ",
            "non_iid",
            ["non_iid", "uniform", "class_distributed"],
        )

        if args_dict["data_dist"] == "class_distributed":
            args_dict["number_of_clients"] = 5
            print(
                emoji.emojize(
                    "\n:exclamation: Number of clients is set as 5 because you chose class_distributed.",
                    use_aliases=True,
                ),
                end="\n\n",
            )
        else:
            args_dict["number_of_clients"] = validate_type_input(
                "Number of clients. ", 10, int
            )

        args_dict["number_of_clients_per_round"] = validate_type_input(
            "Number of participating clients per round. ", 5, int
        )

        if args_dict["number_of_clients_per_round"] > args_dict["number_of_clients"]:
            raise ValueError(
                emoji.emojize(
                    "Number of participating clients can't be larger than the total number of clients. Try again. :x:",
                    use_aliases=True,
                )
            )

        args_dict["seed"] = validate_type_input(f"Random seed. ", None, int)

        print_training_config(args_dict)
        args.output = output

        training_time = federated_pipeline(
            name=args.experiment_name,
            aggregation_method=args_dict.get("aggregation_method"),
            client_weighting=args_dict.get("client_weighting"),
            keras_model_fn=args.model,
            server_optimizer_fn=args.optimizer,
            server_optimizer_lr=args.learning_rate,
            client_optimizer_fn=args_dict.get("client_optimizer"),
            client_optimizer_lr=args_dict.get("client_lr"),
            data_selector=args_dict.get("data_dist"),
            output=args.output,
            client_epochs=args_dict.get("client_epochs"),
            batch_size=args.batch_size,
            number_of_clients=args_dict.get("number_of_clients"),
            number_of_clients_per_round=args_dict.get("number_of_clients_per_round"),
            number_of_rounds=args.epochs,
            iterations=args_dict.get("rfa_iterations"),
            v=args_dict.get("l2_threshold"),
            compression=args_dict.get("compression"),
            dp=args_dict.get("differentially_privacy"),
            noise_multiplier=args_dict.get("noise_multiplier"),
            clipping_value=args_dict.get("clipping_norm"),
            seed=args_dict.get("seed"),
        )

    args_dict["training_time"] = training_time
    path = f"{args.output}/logdir/{args.experiment_name}/training_configuration.json"
    args_dict["output"] = path.replace("/training_configuration.json", "")

    with open(path, "w") as fp:
        json.dump(args_dict, fp, indent=4, sort_keys=True)

    print(
        emoji.emojize(
            f"\nTraining configuration is written to {path} :slightly_smiling_face:",
            use_aliases=True,
        )
    )


if __name__ == "__main__":
    main()
