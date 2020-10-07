======================
Extending this package
======================

This package was designed to be easily extendable and customizable for your own needs. All the technical aspects of this package's extension can be found in the :doc:`API reference<api/api_ref>`. In this page, we only go over how to implement the algorithm itself, but you'll still need to modify the ``predict.py`` file to make the algorithm accessible from the command line.

Adding an sklearn Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Adding a new algorithm to test against is fairly straightforward. The first step is to add a new class to the package. Let's try adding a `SVC algorithm <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC>`_ in a python file called ``svc.py``.

We need to make it implement the ``AlgorithmMeta`` class, which will ensure many builtin functionalities such as ``fit``, ``predict``, or ``score``. The only thing we really need to do now is implement the class's constructor.
::
    """svc.py"""
    from mnist_classifier.algorithm_meta import AlgorithmMeta
    from sklearn.svm import SVC


    class SVCExtension(AlgorithmMeta):

        def __init__(self,
                     random_seed: int=None,         # for reproducibility
                     report_directory: str = None,  # for AlgorithmMeta constructor
                     test_suite_iter: int = None,   # for AlgorithmMeta constructor
                     **svc_configuration_settings): # custom SVC settings

            # initialize parent class
            super().__init__(report_directory, test_suite_iter=test_suite_iter)

            # initialize the model
            self.model = SVC(random_state=random_seed, **svc_configuration_settings)


And that's it! If you wish to have more custom functions (such as ``print_results`` or ``display_results``), feel free to overload any of ``AlgorithmMeta``'s methods.

Adding a Custom Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to design your own algorithm and use this framework to test it, or if you want to implement a non-sklearn algorithm, you definitely can.

We also create a class inheriting from `AlgorithmMeta`, but this time, we have to overload a few functions, since many functions take advantage of the fact that all we used so far was well structured ``sklearn`` classifiers which have similar methods. Namely:

 - ``__init__()`` obviously.
 - ``fit()`` to train the algorithm
 - ``predict()`` to run a prediction
 - ``score()`` which is actually inherited by ``ClassifierMixin``
 - ``save() and load()`` depending on the library you are using.

And this should ensure smooth sailing.