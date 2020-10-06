"""
Handles everything concerning reporting.
"""
import json
import os


def load_test_suite_conf(filepath: str):
    """
    Loads a test suite json configuration file, and returns an array of parameters to pass to the argument parser.
    The JSON file should be formatted like the `test_suite_example.json` file in this repository. The names of the
    dict keys for each test should be

    Parameters
    ----------
    filepath : str
        the filepath to the test suite JSON.

    Returns
    -------
    list
        a list of parameter lists corresponding to the configurations of each of the tests to be run
    """
    output = []
    with open(filepath, "r") as file:
        to_parse = json.load(file)
        file.close()

    if "report_directory" not in to_parse:
        print("You need to specify the report_directory for a test suite")
        raise ValueError

    if "tests" not in to_parse:
        print("Did not find the \"tests\" object in your file. You need a test suite!")
        raise ValueError

    report_dir = prepare_report_dest(to_parse['report_directory'])

    for test in to_parse['tests']:
        global_args = ["--report_directory", report_dir]
        model = []
        model_args = []

        for key in test:
            if key == "type":
                model = [test[key]]
            elif key == "report_directory":
                continue
            elif key in ["load", "save", "random_seed"]:
                global_args.extend(["--" + key, str(test[key])])
            elif key == "hidden_layers":
                model_args.extend([f"--hidden_layers={test[key]}"])
            elif key == "verbose":
                model_args.append("-v")
            else:
                model_args.extend(["--" + key, str(test[key])])
        output.append(global_args + model + model_args)
    return output


def prepare_report_dest(report_filepath: str):
    """
    Prepares the destination output file. Checks if the location exists already, if it does, it creates a unique version
    with an auto_increment. So for example if inputted "my_report" and a folder "my_report" exists, a folder will be
    created called "my_report_1"

    Parameters
    ----------
    report_filepath : str
        the target folder where the report should be created

    Returns
    -------
    str
        the actual filepath that was created
    """
    if not os.path.exists("reports"):
        os.mkdir("reports")

    target = os.path.join("reports", report_filepath)
    if not os.path.exists(target):
        os.mkdir(target)
        return target
    increment = 1
    while os.path.exists(target + f"_{increment}"):
        increment += 1
    target = target + f"_{increment}"
    os.mkdir(target)
    return target
