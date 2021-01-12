from setuptools import setup, find_packages
import os
import re

TEST_REQUIRES = ["pytest"]

DEV_REQUIRES = TEST_REQUIRES + [
    "black",
    "flake8",
    "sphinx",
    "sphinx-autodoc-typehints",
    "sphinx-rtd-theme",
    "sphinxcontrib-spelling",
    "codecov",
]

classifiers = [
    "Development Status :: 3 - Alpha",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.5",
    "License :: OSI Approved :: MIT License",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "Operating System :: OS Independent",
]

# Get the long description from the README file
with open("README.rst", "r", encoding="utf8") as fh:
    long_description = fh.read()

# Get version string from module
init_path = os.path.join(os.path.dirname(__file__), "geotorch/__init__.py")
with open(init_path, "r", encoding="utf8") as f:
    version = re.search(r"__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M).group(1)

setup(
    name="geotorch",
    version=version,
    description="Constrained Optimization and Manifold Optimization in Pytorch",
    author="Mario Lezcano Casado",
    author_email="lezcano-93@hotmail.com",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/Lezcano/geotorch",
    classifiers=classifiers,
    keywords=["Constrained Optimization", "Optimization on Manifolds", "Pytorch"],
    packages=find_packages(),
    python_requires=">=3.5",
    install_requires=["torch>=1.5"],
    extras_require={"dev": DEV_REQUIRES, "test": TEST_REQUIRES},
)
