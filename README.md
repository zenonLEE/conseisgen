# ConSeisGen: Controllable Synthetic Seismic Waveform Generation

<p align="center">
  <img src="images/archi.png" alt="ConSeisGen Architecture" width="700"/>
</p>

<p align="center">
  <a href="https://ieeexplore.ieee.org/document/10339272">
    <img src="https://img.shields.io/badge/View%20on-IEEE%20Xplore-blue" alt="View on IEEE Xplore">
  </a>
</p>

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.9%2B-%23ee4c2c.svg)](https://pytorch.org/)

This repository contains the official implementation of **ConSeisGen**, a controllable generative adversarial network (GAN) framework for generating realistic seismic waveforms conditioned on a continuous physical parameter such as epicentral distance.

> **Paper Reference:**  
> **Li, Y., Yoon, D., Ku, B., & Ko, H. (2023).**  
> *ConSeisGen: Controllable Synthetic Seismic Waveform Generation.* IEEE Geoscience and Remote Sensing Letters, 21, 1–5.  
> [IEEE Link (DOI: 10.1109/LGRS.2023.3342801)](https://doi.org/10.1109/LGRS.2023.3342801)


---

## 📁 Project Structure

```bash
ConSeisGen/
├── configs/         # YAML or JSON files for experiment configs
├── logs/            # Training logs
├── outputs/         # Generated waveform outputs, model checkpoints
├── networks.py      # Model definitions (Generator, Discriminator, blocks)
├── train.py         # Training entry point
├── inference.py     # Inference code
├── trainer.py       # Training loops and loss functions
└── utils.py         # Dataset loading, metrics, and utility functions
```

---

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/ConSeisGen.git
cd ConSeisGen
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

Run the training with a configuration file:
```bash
python train.py --config configs/default.yaml
```

Generated samples and model checkpoints will be saved in `outputs/`.

---

## 🧠 Model Overview

The generator `SeisGen_ACGAN_real_dist` takes as input:
- A random noise vector (e.g. Gaussian)
- A continuous condition (e.g., distance in km)

It outputs:
- A waveform tensor of shape `(batch_size, 3, sequence_length)`

The discriminator predicts both:
- Real/Fake label (adversarial output)
- The input condition (auxiliary regression)

---

## 📊 Results

The proposed method demonstrates improved control over synthetic waveform properties while preserving realism, outperforming conventional GANs in both visual and statistical metrics. See the full paper for quantitative evaluations.

---

## 📷 Sample Output

Below is an example of a generated synthetic seismic waveform using the ConSeisGen model:

<p align="center">
  <img src="/images/sample.png" alt="Sample generated waveform" width="600"/>
</p>

---

## 📝 Citation
If you use this work in your research, please cite:

```bibtex
@article{li2023conseisgen,
  title={ConSeisGen: Controllable Synthetic Seismic Waveform Generation},
  author={Li, Yuanming and Yoon, Doyoung and Ku, Byoungjoo and Ko, Hanseok},
  journal={IEEE Geoscience and Remote Sensing Letters},
  volume={21},
  pages={1--5},
  year={2023},
  publisher={IEEE}
}
```

---

## 📬 Contact
For questions or collaborations, feel free to contact:
- Yuanming Li: [lym7499500@gmail.com](mailto:lym7499500@gmail.com)
