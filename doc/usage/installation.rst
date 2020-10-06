============
Requirements
============

Before you begin using this package, you must first make sure you have python3, conda, and pip installed.
The instructions to get python3 and conda up and running can be found
`on the conda website <https://docs.conda.io/projects/conda/en/latest/user-guide/install/>`_. And then you can just run
::
    conda install pip

============
Installation
============

Using pip
---------

The simplest way to install this project is to do so with pip:
::
    pip install mnist-classifier


Using git
---------

First, clone the repository wherever you wish to use this project::

    cd /path/to/your/directory
    git clone https://github.com/sandrich/classifying_digits_mnist.git

Then you can run the setup script:
::
    python setup.py install

For developers
--------------
If you want to develop and improve this package, you can create a conda environment with the requirements
::
    conda env create --file environment.yml

You can then call the script locally:
::
    # this will print the help menu
    python mnist_classifier/predict.py --help