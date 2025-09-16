# Möller et al. 2025 — The loss of safe land on atoll islands under different emissions pathways

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

## Description

This repository provides Jupyter notebooks and scripts for the pre-processing of input datasets and the application of the **BEWARE framework** to estimate wave-driven flooding on atoll islands, as presented in **Möller et al. (2025)**.  

The repository uses the environment manager [pixi](https://pixi.sh/latest) to ensure reproducibility and compatibility of required packages. Transects used in the study are included under `data/Shapefiles`.

---

## Quick start

```sh
git clone https://gitlab.com/TessaM97/atoll-slr-paper
cd atoll-slr-paper
make virtual-environment
pixi run jupyter lab
```
---

## Installation of python environment

The environment management uses [pixi](https://pixi.sh/latest).
To get started, you will need to make sure that pixi is installed ([instructions here](https://pixi.sh/latest)).

To create the virtual environment, run

```sh
pixi install
pixi run pre-commit install
```

These steps are also captured in the `Makefile` so if you want a single
command, you can instead simply run `make virtual-environment`.

Having installed your virtual environment, you can now run commands in your
virtual environment using

```sh
pixi run <command>
```

For example, to run Python within the virtual environment, run

```sh
pixi run python
```

As another example, to run a notebook server, run

```sh
pixi run jupyter lab
```

## Download of required data

Necessary datasets for reproduction of results include **COWCLIP**, **AR6_SLR_projection**, **COAST_RP**, **BEWARE_database**, as well as the **atoll transects**.

1) Create a folder for data, e.g. `atoll_slr_data`.
2) Update `DATA_DIR` in `src/settings.py` to point to this folder.
2) Execute `100_download_BEWARE_database`,`101_download_COWCLIP` , `102_download_AR6_SLR_projections`, `103_download_COAST_RP` under `notebooks/additionnal_notebooks`. This will download the necessary external datasets.

Example:
```sh
pixi run python notebooks/additional_notebooks/101_download_COWCLIP.py
```

## Running the analysis

Running the analysis
The main workflow is organized into sequentially numbered notebooks:
`010–020`: Data preprocessing
`030`: Matching transects with the best BEWARE database entries
Follow them in numerical order. Each notebook contains detailed explanations.


## License

This repository, including all data and scripts, is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) license. Please refer to the individual license texts in the LICENSE file for more details.

## References

If you use this repository, please cite:
Möller, T., et al. (2025). The loss of safe land on atoll islands under different emissions pathways. [Journal Name], [DOI link]

Please also cite any of the external datasets that you use (BEWARE_database, COWCLIP, AR6_SLR_projection COAST_RP,...)

## Contact

For questions, please contact: moeller@iiasa.ac.at

### Original template

This repository was generated from this template:
[basic python repository](https://gitlab.com/openscm/copier-basic-python-repository). [copier](https://copier.readthedocs.io/en/stable/) is used to manage and distribute this template.