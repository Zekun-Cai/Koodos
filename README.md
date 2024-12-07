<h1 align="center">[NeurIPS 24] Continuous Temporal Domain Generalization</h1>

</div>

<div align="center">

<a href="https://arxiv.org/abs/2405.16075"><img alt="Static Badge" src="https://img.shields.io/badge/PDF-Arxiv-red?style=flat&color=%23b12221"></a>
<a href="https://neurips.cc/virtual/2024/poster/95933"><img alt="Static Badge" src="https://img.shields.io/badge/Pub-NeurIPS24-purple?style=flat&color=purple"></a>
<a href="https://openreview.net/forum?id=G24fOpC3JE"><img alt="Static Badge" src="https://img.shields.io/badge/Review-OpenReview-blue?style=flat&color=%230077cc"></a>
<a href="https://github.com/Yukayo/LoTNext/blob/main/LICENSE"><img alt="Static Badge" src="https://img.shields.io/badge/License-MIT-orange?style=flat"></a>
<img src="https://badges.toozhao.com/badges/01JEGVGX082SC75NK1KYHDX15W/orange.svg" alt="Static Badge" />

**[<a href="https://mp.weixin.qq.com/s/c2JjCB5bxRU7NgCELPLsdQ">中文解读</a>]**
</div>

This repository contains the official code and datasets for the paper, "Continuous Temporal Domain Generalization", accepted by NeurIPS 2024.

**TL;DR**: The study proposes a method capable of generating applicable neural networks at any given moment, based on observing domain data at random time points within a concept drift environment.

<p align="center">
  <img src="./figures/concept_1080.gif" width="600">
</p>

## Abstract
Temporal Domain Generalization (TDG) traditionally deals with training models on temporal domains collected at fixed intervals, limiting their ability to handle continuously evolving, irregular temporal domains. This work introduces Continuous Temporal Domain Generalization (CTDG) and presents Koodos, a model designed to address and optimize this challenge. 

Koodos comprises three key components: 1) Describing the evolution of model parameters by constructing a dynamical system; 2) Modeling complex nonlinear dynamics by Koopman Theory; and 3) Joint optimization of the model and its dynamics.

<p align="center">
  <img src="./figures/Illustration.png" width="800">
</p>

## Table of Contents
1. [Installation](#installation)
2. [Quick Demo](#quick-demo)
3. [Usage](#usage)
4. [Code Structure](#code-structure)
5. [Citation](#citation)
6. [Further Reading](#further-reading)

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

Before diving into the source code, we provide a Jupyter Notebook [Tutorial_for_Koodos](./Tutorial_for_Koodos.ipynb)  in this repository, **which provides a step-by-step guide to the Koodos framework and demonstrates its core functionality on a sample dataset to give you an intuitive understanding of how it works.**

<p align="center">
  <img src="./figures/demo.gif" width="800">
</p>

## Usage

To train and test the model using the Koodos framework, follow this command:

```bash
python main.py --dataset <dataset-name> --cuda <GPU-No.>
```

Available Datasets: Moons; MNIST; Twitter; YearBook; Cyclone; House.

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
@inproceedings{cai2024continuous,
  title={Continuous Temporal Domain Generalization},
  author={Cai, Zekun and Bai, Guangji and Jiang, Renhe and Song, Xuan and Zhao, Liang},
  booktitle = {Advances in Neural Information Processing Systems},
  year={2024}
}
```

## Related Reading
[Temporal Domain Generalization with Drift-Aware Dynamic Neural Networks](https://openreview.net/pdf?id=sWOsRj4nT1n), in ICLR 2023.  
[MemDA: Forecasting Urban Time Series with Memory-based Drift Adaptation](https://arxiv.org/abs/2309.14216), in CIKM 2023.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
