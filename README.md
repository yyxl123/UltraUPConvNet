# UltraUPConvNet
 [[Preprint on ArXiv](https://arxiv.org/abs/2509.11108)]

 This repository hosts the official PyTorch implementation and pretrained model weights for our paper, "UltraUPConvNet: A UPerNet- and ConvNeXt-Based Multi-Task Network for Ultrasound Tissue Segmentation and Disease Prediction"

## 🎯 About the Challenge

The UUSIC25 challenge aims to spur the development of universal models for ultrasound image analysis. Ultrasound imaging is a cornerstone of biomedical diagnostics, but creating models that perform accurate classification and segmentation across diverse organs and pathologies remains a significant hurdle.

This challenge encourages participants to develop innovative algorithms that can handle multiple tasks (classification and segmentation) on a wide range of ultrasound images from 7 different organs. Our goal is to push the boundaries of model generalization, efficiency, and clinical applicability.

## 🚀 Getting Started

Follow these steps to set up your environment and run the model.

### 1. Clone Repository

```bash
git clone https://github.com/yyxl123/UltraUPConvNet.git

cd UltraUPConvNet
```

### 2. Create Environment

We recommend using Conda to manage the environment.

```bash
# Create a new conda environment
conda create -n us python=3.10 -y
conda activate us

# Install PyTorch (ensure compatibility with your CUDA version)
# Example for CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

A possible `requirements.txt` is provided below for your convenience:
```txt
numpy
opencv-python
yacs
timm
einops
scipy
tqdm
tensorboard
medpy
Pillow
```

### 3. Prepare Datasets

The data structure is crucial for the data loaders to work correctly.

Your final `data` directory structure should look similar to this:

    ```
    data/
    ├── classification
    │   ├── Appendix
    │   ├── BUS-BRA
    │   ... (other public datasets)
    │   ├── private_Appendix
    │   ├── private_Breast
    │   └── ... (other prefixed private datasets)
    ├── segmentation
    │   ├── BUS-BRA
    │   ... (other public datasets)
    │   ├── private_Breast
    │   └── ... (other prefixed private datasets)
    └── Val
        ├── classification
        └── segmentation

2.  **Generate File Lists**: After organizing the data, you need to generate `train.txt`, `val.txt`, and `test.txt` for each dataset. We provide a script to do this automatically.

    ```bash
    python ./datasets/generate_txt.py
    ```
    This script will scan the `data/` directory, split the files into 70% training, 20% validation, and 10% testing sets, and write the file paths into the corresponding `.txt` files.

### 4. Download Pre-trained Weights (Optional)

To get started quickly or if you want to download ConvNeXt weight, you can follow:

1.  **Download Model Weights**: Go to the official [**GitHub Releases page**](https://github.com/facebookresearch/ConvNeXt) and download the `ConvNeXt-T 224*224` file.
2.  **Prepare Experiment Directory**: Create a directory for your experiment output: `mkdir pretrained_ckpt`.
3.  **Place and Rename Weights**: Move the downloaded `*.pth` file into this folder.


## 🏋️‍♀️ Model Training

The training process is handled by `train.py`, which leverages `trainer.py`. It uses a sophisticated weighted sampler to balance learning between different datasets and tasks (segmentation and classification).

To start training, run the provided shell script:

```bash
bash train.sh
```

Alternatively, you can run the command directly. For multi-GPU training (e.g., 2 GPUs):

```bash
python train.py --output_dir=exp_out/prompt \
                --prompt \
                --base_lr=3e-5 \
                --use_pretrained \
                --pretrained_path=./pretrained_ckpt/convnext_tiny_22k_1k_224.pth \
                --batch_size=16

```

**Key Arguments**:
- `--output_dir`: Directory to save logs, checkpoints, and validation results.
- `--prompt`: Enables the prompt-based learning mechanism.
- `--batch_size`: batch size.
- `--max_epochs`: Total number of training epochs.
- `--pretrain_ckpt`: Path to a pretrained ConvNeXt checkpoint (`.pth`) to initialize the encoder. The baseline will automatically load from `pretrained_ckpt/convnext_tiny_22k_1k_224.pth`.

Checkpoints and logs will be saved in the specified `--output_dir`.

## 🧪 Inference and Evaluation

After training, you can evaluate your model on the test sets using `test.py`.

To run evaluation on a single GPU:

```bash
python  test.py     --output_dir=exp_out/prompt     --prompt     --is_saveout
```

## ©️ Citation

If you use this baseline or find our work helpful, please consider citing:

```bibtex
@misc{chen2025ultraupconvnetupernetconvnextbasedmultitask,
      title={UltraUPConvNet: A UPerNet- and ConvNeXt-Based Multi-Task Network for Ultrasound Tissue Segmentation and Disease Prediction}, 
      author={Zhi Chen and Le Zhang},
      year={2025},
      eprint={2509.11108},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2509.11108}, 
}
# Please also add a citation for the UUSIC25 challenge paper once it is available.
```
