This folder contains the documentation of GeoTorch.

To make the documentation you can run

```
make html
```

To generate the docs from the GeoTorch source excluding the `parametrize.py` file, run

```
sphinx-apidoc -o ./source ../geotorch ../geotorch/parametrize.py
```
