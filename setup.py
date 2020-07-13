#!/usr/bin/env python3

from setuptools import setup, find_packages
import sys
import os
import re

REQUIRED_MAJOR = 3
REQUIRED_MINOR = 5

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


TEST_REQUIRES = ["pytest"]

DEV_REQUIRES = TEST_REQUIRES + [
    "black",
    "flake8",
    "sphinx",
    "sphinx-autodoc-typehints",
    "sphinx-rtd-theme",
    "codecov",
]


# Get the long description from the README file
with open("README.rst", "r", encoding="utf8") as fh:
    long_description = fh.read()

# get version string from module
init_path = os.path.join(os.path.dirname(__file__), "geotorch/__init__.py")
with open(init_path, "r", encoding="utf8") as f:
    version = re.search(r"__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M).group(1)

setup(
    name="geotorch",
    version=version,
    description="Constrained Optimization and Manifold Optimization in Pytorch",
    author="Mario Lezcano Casado",
    license="MIT",
    long_description=long_description,
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
    python_requires=">={}.{}".format(REQUIRED_MAJOR, REQUIRED_MINOR),
    install_requires=["torch>=1.5"],
    extras_require={"dev": DEV_REQUIRES, "test": TEST_REQUIRES},
)
