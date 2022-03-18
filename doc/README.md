# RLlib Configurators Documentation


## Installation

To build the documentation, make sure you have `ray` installed first.
For building the documentation locally install the following dependencies:

```bash
pip install -r requirements-doc.txt
```

## Building the documentation

To compile the documentation and open it locally, run the following command from this directory.

```bash
sphinx-build -W -b html -d _build/doctrees  source _build/html && open _build/html/index.html
```
