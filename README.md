<h1 align="center">[NeurIPS 24] Continuous Temporal Domain Generalization</h1>

This repository contains the official code and datasets for the paper, "Continuous Temporal Domain Generalization", accepted by NeurIPS 2024.

Paper can be found here: [![Arxiv link](https://img.shields.io/static/v1?label=arXiv&message=CTDG&color=red&logo=arxiv)](https://arxiv.org/abs/2405.16075)

## Abstract
Temporal Domain Generalization (TDG) traditionally deals with training models on temporal domains collected at fixed intervals, limiting their ability to handle continuously evolving, irregular temporal domains. This work introduces Continuous Temporal Domain Generalization (CTDG) and presents Koodos, a model designed to address and optimize this challenge. 

Koodos comprises three key components: 1) Describing the evolution of model parameters by constructing a dynamical system; 2) Modeling complex nonlinear dynamics by Koopman Theory; and 3) Joint optimization of the model and its dynamics.

The code and instructions for reproducing the results are provided in this repository.

<p align="center">
  <img src="./figures/Illustration.png" width="790">
</p>

## Table of Contents
1. [Installation](#installation)
2. [Quick Demo](#quick_demo)
3. [Usage](#usage)
4. [Code Structure](#code-structure)
5. [Citation](#citation)
6. [Further Reading](#further_reading)

---

## Installation

To get started, clone the repository and install the required dependencies.

```bash
git clone https://github.com/Zekun-Cai/Koodos.git
cd Koodos
pip install -r requirements.txt
```

### Downloading the Dataset
The datasets used in this project can be downloaded from [Dataset Download Link](https://drive.google.com/drive/folders/1_w4_H-A8qW5Os6ZT2jidLYAa2weFuX1r?usp=sharing). After downloading, please place the files in the ```data``` directory.

## Quick Demo

Before diving into the source code, we provide a Jupyter Notebook tutorial [Tutorial_for_Koodos](./Tutorial_for_Koodos.ipynb)  in this repository. It offers a step-by-step guide to understanding the Koodos framework, demonstrating its core functionality on a sample dataset.

**This tutorial will walk you through the basic setup and provide an intuitive understanding of how Koodos works.**

## Usage

To train and test the model using the Koodos framework, follow this command:

```bash
python main.py --dataset <dataset-name> --cuda <GPU-No.>
```

You can modify the hyperparameters by adjusting the ```param.py``` config file.

## Code Structure

```
|-- data/                # Dataset Files
|-- model/               # Dataset-specific Model Architectures (e.g., Moons, MNIST)
|-- save/                # Folder to Store Model Outputs and Logs
|-- param.py             # Configuration Files for Different Datasets
|-- util.py              # Utility Functions
|-- koodos.py            # Definition for Koodos
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

## Further Reading
[Temporal Domain Generalization with Drift-Aware Dynamic Neural Networks](https://openreview.net/pdf?id=sWOsRj4nT1n), in ICLR 2023.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
