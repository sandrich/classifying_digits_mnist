"""
Package setup
"""
#!/usr/bin/env python
# coding=utf-8

import os
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install

# circleci.py version
VERSION = "1.0.4"

def load_requirements(file):
    """
    Loads requirements file
    """
    retval = [str(k.strip()) for k in open(file, "rt")]
    return [k for k in retval if k and k[0] not in ("#", "-")]

class VerifyVersionCommand(install):
    """Custom command to verify that the git tag matches our version"""
    description = 'verify that the git tag matches our version'

    def run(self):
        tag = os.getenv('CIRCLE_TAG')

        if tag != VERSION:
            info = "Git tag: {0} does not match the version of this app: {1}".format(
                tag, VERSION
            )
            sys.exit(info)


setup(
    name="mnist-classifier",
    version=VERSION,
    description="Basic mnist classifier example of a Reproducible Research Project in Python",
    url="https://github.com/sandrich/classifying_digits_mnist",
    license="MIT",
    author="Christian Sandrini",
    author_email="mail@chrissandrini.ch",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    install_requires=load_requirements("requirements.txt"),
    entry_points={"console_scripts": ["mnist-predict = mnist_classifier.predict:main"]},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    cmdclass={
        'verify': VerifyVersionCommand,
    }
)
