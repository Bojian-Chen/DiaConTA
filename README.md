# DiaConTA

A Domain Incremental Continual Test-Time Adaptation framework for fault diagnosis and signal classification tasks.

## 📖 Overview

DiaConTA is a PyTorch-based framework designed for continual test-time adaptation in changing domain scenarios. The framework addresses the challenge of adapting pre-trained models to sequential domain shifts commonly encountered in real-world industrial applications, such as bearing fault diagnosis, gearbox monitoring, and acoustic signal classification.

## ✨ Key Features

- **Multiple Adaptation Methods**: Support for various test-time adaptation approaches including:
  - Baseline methods:  Norm, PseudoLabel, Tent, SHOT
  - Advanced methods: CoTTA, GSFDA, UCSN, CoSDA, EATA, RaTP
  - Custom methods: `ours`, `ours_simple`, `ours_new`, `ours_sc`

- **Flexible Backbone Networks**:
  - ResNet14, ResNet32 (for 2D frequency domain data)
  - ResNet18-1D, CNN (for 1D time-series data)

- **Multiple Classification Heads**:
  - Fully Connected (FC)
  - Cosine Similarity Classifier
  - Euclidean Distance Classifier
  - Weight Normalized FC

- **Rich Data Augmentation**:
  - Label Smoothing
  - Prototypical Contrastive Learning (PCL)
  - RandMix augmentation
  - Contrastive loss options

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision numpy scipy argparse
```

### Dataset Support

The framework supports multiple industrial fault diagnosis datasets:

- **SK**:  SEU bearing dataset (10 classes, 24 domains)
- **iFlytek**: Acoustic signal dataset (5 classes, multiple domains)
- **WT**: Wind turbine dataset (5 classes, 5 domains)
- **PU_Real**:  Paderborn University real bearing data (5 classes, 4 domains)
- **PU_Art**: Paderborn University artificial fault data (8 classes, 4 domains)
- **HUST**:  HUST bearing dataset (9 classes, 10 domains)
- **Robot**: Robot dataset (4 classes, 4 domains)

### Data Format

Place your dataset files in the `./data/` directory. Data should be in MATLAB `.mat` format with appropriate structure. 

## 🔧 Usage

### Basic Training

```bash
python main.py --dataset_name SK \
               --backbone_name resnet14 \
               --incremental_mode ours \
               --base_epochs 40 \
               --epochs 40 \
               --batch_size 64
```

### Advanced Options

**Select Adaptation Method:**
```bash
# Use custom method with all components
python main.py --incremental_mode ours

# Use simplified method with ablation options
python main.py --incremental_mode ours_simple --TOPK --PCA --MI --SR

# Use baseline methods
python main.py --incremental_mode Tent
```

**Configure Training:**
```bash
python main.py --dataset_name iFlytek \
               --backbone_name resnet14 \
               --classifer cos \
               --base_lr 0.1 \
               --lr 0.1 \
               --base_epochs 40 \
               --epochs 40 \
               --batch_size 64 \
               --preprocess zscore \
               --eta_min 0.001
```

**Enable Augmentations:**
```bash
python main.py --contrastive_loss \
               --LabelSmooth \
               --PCL \
               --RandMix
```

## 📊 Arguments

### Basic Parameters
- `--random_seed`: Random seed for reproducibility (default: 2024)
- `--backbone_name`: Backbone network architecture
- `--classifer`: Type of classifier head
- `--preprocess`: Data preprocessing method (zscore/minmax/None)

### Training Parameters
- `--base_epochs`: Number of epochs for source training (default: 40)
- `--base_lr`: Learning rate for source training (default: 0.1)
- `--epochs`: Number of epochs for adaptation (default: 40)
- `--lr`: Learning rate for adaptation (default: 0.1)
- `--batch_size`: Batch size for training (default: 64)

### Adaptation Methods
- `--incremental_mode`: Choose from multiple adaptation strategies
  - `norm`: No adaptation (baseline)
  - `PesudoLabel`: Pseudo-labeling approach
  - `Tent`: Test-time normalization
  - `shot`: SHOT method
  - `CoTTA`: Continual TTA
  - `gsfda`: GSFDA method
  - `UCSN`: UCSN method
  - `CoSDA`: CoSDA method
  - `EATA`: EATA method
  - `RaTP`: RaTP method
  - `ours`: Proposed method (full version)
  - `ours_simple`: Proposed method (simplified)
  - `ours_new`: Proposed method (new variant)

### Ablation Study Options
- `--TOPK`: Enable top-k selection
- `--PCA`: Enable PCA-based selection
- `--MI`: Enable mutual information
- `--SR`: Enable sample reweighting
- `--mixup`: Enable mixup augmentation

## 📁 Project Structure

```
DiaConTA/
├── data/                   # Dataset directory
├── log/                    # Training logs
├── models/                 # Model architectures
├── trainer/                # Training logic
��── utils/                  # Utility functions
│   ├── set.py              # Random seed setting
│   └── eval_metric.py      # Evaluation metrics
├── main.py                 # Main training script
├── dataloader_domain.py    # Domain-specific data loaders
├── loss_function.py        # Custom loss functions
└── README.md               # This file
```

## 🎯 Model Saving

Models are automatically saved by default.  To disable model saving: 
```bash
python main.py --save_model
```

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{diaconta2026,
  author = {Chen, Bojian},
  title = {DiaConTA: Domain Incremental Continual Test-Time Adaptation},
  year = {2026},
  url = {https://github.com/Bojian-Chen/DiaConTA}
}
```

## 👤 Author

**Bojian Chen**

## 📄 License

This project is open source and available under the terms specified in the repository.

## 🙏 Acknowledgments

This framework builds upon various test-time adaptation methods and incorporates techniques from the domain adaptation and continual learning communities. 

---

**Note**: This is a research project under active development. For questions or collaboration opportunities, please open an issue in the repository.