"""
Handles everything concerning reporting.
"""
import os


def load_conf_file(filepath: str):
    pass


def prepare_report_dest(report_filepath: str, force: bool = False):
    """
    Prepares the destination output file. Checks if the location exists already, if it does, it creates a unique version
    with an auto_increment. So for example if inputted "my_report" and a folder "my_report" exists, a folder will be
    created called "my_report_1"

    Parameters
    ----------
    report_filepath : str
        the target folder where the report should be created
    force : bool
        if True, does not check if the folder already exists, and will overwrite

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
    elif force:
        return target
    else:
        increment = 1
        while os.path.exists(target + f"_{increment}"):
            increment += 1
        target = target + f"_{increment}"
        os.mkdir(target)
        return target
