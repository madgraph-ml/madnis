<p align="center">
  <img src="docs/source/_static/logo-light.png" width="450", alt="MadNIS">
</p>

<h2 align="center">Neural Multi-Channel Importance Sampling</h2>

<p align="center">
<img alt="Build Status" src="https://github.com/madgraph-ml/madnis/actions/workflows/CI.yml/badge.svg">
<a href="https://arxiv.org/abs/2212.06172"><img alt="Arxiv" src="https://img.shields.io/badge/arXiv-2212.06172-b31b1b.svg"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://pytorch.org"><img alt="pytorch" src="https://img.shields.io/badge/PyTorch-2.0-DC583A.svg?style=flat&logo=pytorch"></a>
</p>

This a machine learning framework to perform neural multi-channel importance sampling in MadGraph5.
It containes modules to construct a machine-learning based Monte Carlo integrator using PyTorch.


## Installation

```sh
# clone the repository
git clone https://github.com/madgraph-ml/madnis.git
# then install in dev mode
cd madnis
pip install --editable .
```

## Citation

If you use this code or parts of it, please cite:

    @article{Heimel:2023ngj,
      author = "Heimel, Theo and Huetsch, Nathan and Maltoni, Fabio and Mattelaer, Olivier and Plehn, Tilman and Winterhalder, Ramon",
      title = "{The MadNIS reloaded}",
      eprint = "2311.01548",
      archivePrefix = "arXiv",
      primaryClass = "hep-ph",
      reportNumber = "IRMP-CP3-23-56, MCNET-23-12",
      doi = "10.21468/SciPostPhys.17.1.023",
      journal = "SciPost Phys.",
      volume = "17",
      number = "1",
      pages = "023",
      year = "2024"}
