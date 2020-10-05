====================
Algorithm Parameters
====================

Common parameters
=================
All algorithms share two settings:

Save model
^^^^^^^^^^

``--save | -s``

Enter a filename for the model to save its trained state once it has finished training. It can be used for future use.
Defaults to None
::
    python mnist_classifier.py rf --save rf.model


Load model
^^^^^^^^^^
``--load | -l``

Enter a filename of an **existing** model which you have previously saved. Defaults to None
::
    python mnist_classifier.py rf --load rf.model


.. NOTE::
    If you specified settings for your training and load a model, the loaded model's settings will override your specified settings.

-----------

Random Forest Parameters
========================

Number of trees / estimators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``--trees | -t``

Defines the number of trees the algorithm generates. Defaults to 20
::
    python mnist_classifier.py rf --trees 10

Tree depth
^^^^^^^^^^
``--depth | -d``

Defines how deep we allow the trees to grow. Defaults to 9
::
    python mnist_classifier.py rf --depth 50

Impurity method
^^^^^^^^^^^^^^^
``--impurity_method | -i``

Defines which impurity method to use when deciding which parameter to select for a split.
Can either be ``gini`` or ``entropy``, Defaults to ``entropy``
::
    python mnist_classifier rf -i gini

-------------

MLP Parameters
==============

Hidden layers
^^^^^^^^^^^^
``--hidden_layers | -hl``

Defines how many hidden layers, and how many neurons per hidden layer to construct.
Defaults to one hidden layer of 100 neurons
For example, to build a model with three layers : 100 in the first, 25 in the second, and 15 in the third, you would use:
::
    python mnist_classifier.py mlp -hl 100 25 15

Alpha bias
^^^^^^^^^^
``--alpha | -a``

Defines the alpha bias of the MLP. Defaults to 0.0001
::
    python mnist_classifier.py mlp -a 0.01

Batch Size
^^^^^^^^^^
``--batch_size | -b``

Defines the size of the training batches. Defaults to 200
::
    python mnist_classifier.py mlp -b 50

Maximum iterations
^^^^^^^^^^^^^^^^^^
``--max_iter | -i``

Defines how many training iterations the algorithm should run before stopping training.
Note that the algorithm might not do exactly this many iterations if no improvement is seen after several iterations.
Defaults to 200.
::
    python mnist_classifier.py mlp -i 10

Verbose mode
^^^^^^^^^^^^
``--verbose | -v``

Makes the algorithm print out each iteration step.
::
    python mnist_classifier.py mlp -v
