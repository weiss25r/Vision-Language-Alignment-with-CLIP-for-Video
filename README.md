# Vision-Language-Alignment-with-CLIP-for-Video

[![Report](https://img.shields.io/badge/Paper-REPORT.md-blue)](docs/REPORT.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 👥 Group and Project Information
- **Group ID**: Justgood AI 
- **Project ID**: 15
- **Group Members**: Edoardo Tantari, Raffaele Terracino

## 📝 Project Description
Searching for videos traditionally relies on manually curated metadata rather than visual content. This project explores zero-shot cross-modal retrieval by aligning video features with natural language text using a contrastive loss model reminiscent of CLIP.

Project done as part of the course [**Deep Learning - Advanced Models and Methods**](https://antoninofurnari.github.io/deeplearning/) at University of Catania.

> 📖 **Official Report**: For all theoretical details, performance analysis, the architecture used, and group contributions, please refer to our formal paper: **[REPORT.md](docs/REPORT.md)**.

## 🛠 Technical Reproducibility

### 1. Environment and data Setup

**Prerequisites:**
To get started, clone the repo and install required libraries using pip:

```bash
pip install requirements.txt
```

or by using conda:

```bash
conda env create -f environment.yml
conda activate vla
```

**Dataset:**
to replicate the entire pipeline, you'll need to download the [Epic Kitchens dataset](LINK). If you want to only replicate baseline and best model training, you can just download our extracted features from [here](link).

### 2. Training
You can start training using the following commands.

```bash
python src/training/trainer.py --config experiments/configs/experiment.yaml
```
You can also resume training from a checkpoint using ```--ckpt``` and specifying the checkpoint path.
For each experiment, we provided both the "last" checkpoint, correspondind to the last epoch, and the "best" one, corresponding to the lowest validation loss. 

Use file ```MLP_timesformer_config.yaml``` for **baseline training** and ```egovlp_egonceloss_config.yaml``` for **best model training.**
### 3. Evaluation
You can test trained models using the following command:

```bash
python src/evaluation/evaluate.py --config experiments/configs/experiment.yaml --ckpt <path_to_checkpoint> --test
```

To run inference on the validation set, you can use the ```--validate``` flag.
For each experiment a checkpoint is provided, the one corresponding to the lowest validation loss.

---

*For the declaration of individual tasks and the use of AI, refer to `docs/REPORT.md`.*
