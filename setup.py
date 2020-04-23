#!/usr/bin/env python3

from setuptools import setup, find_packages
import sys
from os import path

REQUIRED_MAJOR = 3
REQUIRED_MINOR = 6

# Check for python version
if sys.version_info < (REQUIRED_MAJOR, REQUIRED_MINOR):
    error = (
        "Your version of python ({major}.{minor}) is too old. You need "
        "python >= {required_major}.{required_minor}."
    ).format(
        major=sys.version_info.major,
        minor=sys.version_info.minor,
        required_minor=REQUIRED_MINOR,
        required_major=REQUIRED_MAJOR,
    )
    sys.exit(error)


TEST_REQUIRES = ["pytest", "pytest-cov"]

DEV_REQUIRES = TEST_REQUIRES + ["black", "flake8", "sphinx", "sphinx-autodoc-typehints"]


# Get the long description from the README file
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="geotorch",
    version="0.1.0",
    description="Constrained Optimization in Pytorch",
    author="Mario Lezcano Casado",
    license="MIT",
    long_description=long_description,  # Optional
    long_description_content_type="text/markdown",
    url="https://github.com/Lezcano/geotorch",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
    ],
    keywords=["Constrained Optimization", "Optimization on Manifolds", "Pytorch"],
    packages=find_packages(),
    python_requires=">=3.6,",
    install_requires=["torch>=1.4"],
    extras_require={
        "dev": DEV_REQUIRES,
        "test": TEST_REQUIRES,
    },
)
