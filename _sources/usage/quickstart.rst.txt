===========
Quick start
===========

The script can either run the detection using as a Random Forest, or with an MLP Neural Network.
The simplest "out-of-the-box" way to run the script is to call the script with either the ``rf`` or ``mlp`` flags: ::

    # run the train/test with a Random Forest
    python mnist_classifier.py rf

    # run the train/test with an MLP
    python mnist_classifier.py mlp

Both algorithms have default settings, so no additional input is required.
