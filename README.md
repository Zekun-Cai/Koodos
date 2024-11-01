<h1 align="center">[NeurIPS 24] Continuous Temporal Domain Generalization</h1>

This repository contains the official code and datasets presented in our paper, "Continuous Temporal Domain Generalization", accepted by the Thirty-Eighth Annual Conference on Neural Information Processing Systems (NeurIPS 2024).

For a detailed explanation of the framework, please refer to the paper. [![Arxiv link](https://img.shields.io/static/v1?label=arXiv&message=CTDG&color=red&logo=arxiv)](https://arxiv.org/abs/2405.16075)

## Abstract
Temporal Domain Generalization (TDG) traditionally deals with training models on domains collected at fixed time intervals, limiting their ability to handle continuously evolving, irregular temporal domains. This work introduces Continuous Temporal Domain Generalization (CTDG) and presents Koodos, a model designed to address and optimize this challenge. 

The code and instructions for reproducing the results are provided in this repository.

<p align="center">
  <img src="./figures/Illustration.png" width="790">
</p>

## Table of Contents
1. [Installation](#installation)
2. [Jupyter Tutorial](#jupyter-tutorial)
3. [Usage](#usage)
4. [Code Structure](#code-structure)
5. [Citation](#citation)

---

## Installation

To get started, clone the repository and install the required dependencies.

```bash
git clone https://github.com/Zekun-Cai/Koodos.git
cd Koodos
pip install -r requirements.txt
```

## Jupyter Tutorial

Before diving into the source code, we provide a Jupyter notebook tutorial [Tutorial_for_Moons](./Tutorial_for_Moons.ipynb)  in this repository. It offers a step-by-step guide to understanding the Koodos framework, demonstrating its core functionality on a sample dataset.

**This tutorial will walk you through the basic setup and provide an intuitive understanding of how Koodos works.**

## Usage

To train and test the model using the Koodos framework, follow this command:

```bash
python main.py --dataset <dataset-name> --cuda <GPU-No.>
```

You can modify the hyperparameters and experiment settings by adjusting the ```param.py``` config file.

## Code Structure

```
|-- model/               # Dataset-specific Model Architectures (e.g., Moons, MNIST)
|-- save/                # Folder to Store Model Outputs and Logs
|-- param.py             # Configuration Files for Different Dataset
|-- util.py              # Utility Functions
|-- koodos.py            # Definitions for Koodos
|-- main.py              # Program Entry, Data Loading, Training, and Testing
```

## Citation
If you find our work useful, please cite the following:

```  
@article{cai2024continuous,
  title={Continuous Temporal Domain Generalization},
  author={Cai, Zekun and Bai, Guangji and Jiang, Renhe and Song, Xuan and Zhao, Liang},
  journal={arXiv preprint arXiv:2405.16075},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
