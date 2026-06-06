This folder contains the documentation of GeoTorch.

To make the documentation you can run

```
make html
```

To generate the docs from the GeoTorch source, run

```
SPHINX_APIDOC_OPTIONS=members sphinx-apidoc -o ./source ../geotorch
```

To check the spelling
```
sphinx-build -b spelling docs/source docs/build
```
