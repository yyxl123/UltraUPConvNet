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

To get started quickly or if you want to **skip the training step** and jump directly to inference, you can use our provided baseline weights.

1.  **Download Model Weights**: Go to our official [**GitHub Releases page**](https://github.com/uusic2025/challenge/releases/latest) and download the `baseline.pth` file.
2.  **Prepare Experiment Directory**: Create a directory for your experiment output: `mkdir -p exp_out/trial_1`.
3.  **Place and Rename Weights**: Move the downloaded `baseline.pth` file into this folder and **rename it to `best_model.pth`**. The testing script `omni_test.py` is configured to load the model from `exp_out/trial_1/best_model.pth`.
4.  **(Recommended) Prepare Backbone Checkpoint**: The code may try to load a pre-trained Swin Transformer backbone. To prevent potential errors during initialization, create a `pretrained_ckpt` folder in the project root and place the `swin_tiny_patch4_window7_224.pth` file inside it. You can download this from the official Swin Transformer repository. This step is only for initializing the encoder's weights and is separate from loading our fine-tuned `best_model.pth`.

Now you can proceed directly to the [Inference and Evaluation](#-inference-and-evaluation) section using this model.

## 🏋️‍♀️ Model Training

The training process is handled by `omni_train.py`, which leverages `omni_trainer.py`. It uses a sophisticated weighted sampler to balance learning between different datasets and tasks (segmentation and classification).

To start training, run the provided shell script:

```bash
bash baseline.sh
```

Alternatively, you can run the command directly. For multi-GPU training (e.g., 2 GPUs):

```bash
python -m torch.distributed.launch \
    --use_env \
    --nproc_per_node=2 \
    --master_port=12345 \
    omni_train.py \
    --output_dir=exp_out/trial_1 \
    --prompt \
    --base_lr=0.003 \
    --batch_size=32 \
    --max_epochs=200
```

**Key Arguments**:
- `--output_dir`: Directory to save logs, checkpoints, and validation results.
- `--prompt`: Enables the prompt-based learning mechanism.
- `--batch_size`: Total batch size across all GPUs.
- `--max_epochs`: Total number of training epochs.
- `--pretrain_ckpt`: Path to a pretrained Swin Transformer checkpoint (`.pth`) to initialize the encoder. The baseline will automatically load from `pretrained_ckpt/swin_tiny_patch4_window7_224.pth`.

Checkpoints and logs will be saved in the specified `--output_dir`. The best-performing model on the validation set will be saved as `best_model.pth`.

## 🧪 Inference and Evaluation

After training, you can evaluate your model on the test sets using `omni_test.py`.

To run evaluation on a single GPU:

```bash
python -m torch.distributed.launch \
    --use_env \
    --nproc_per_node=1 \
    --master_port=12345 \
    omni_test.py \
    --output_dir=exp_out/trial_1 \
    --prompt \
    --is_saveout
```
