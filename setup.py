#!/usr/bin/env python
# coding=utf-8

from setuptools import setup, find_packages


def load_requirements(f):
    retval = [str(k.strip()) for k in open(f, "rt")]
    return [k for k in retval if k and k[0] not in ("#", "-")]


setup(
    name="mnist-classifier",
    version="1.0.0",
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
)