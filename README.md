# EAST

PyTorch implementation of **Spatio-temporal feature learning for enhancing video quality based on screen content characteristics**.

**Ziyin Huang, Yui-Lam Chan, Sik-Ho Tsang, Ngai-Wing Kwong, Kin-Man Lam, and Wing-Kuen Ling**

*Journal of Visual Communication and Image Representation*, Volume 104, Article 104270, 2024.

[Paper](https://doi.org/10.1016/j.jvcir.2024.104270)

## Overview

EAST (edge aware with spatio-temporal information fusion network) is an alignment-free method for enhancing compressed screen content videos. It combines information from neighboring frames with target-frame edges to address scene switches and frame freezing.

The model takes seven consecutive luminance frames and enhances the center frame. Three branches extract features from a seven-frame group and two four-frame groups using 3D convolutions. An edge-aware block uses Sobel features from the target frame, while a U-shaped fusion network and channel/spatial attention combine the extracted features. The reconstructed residual is added to the compressed center frame.

This repository includes the 48-channel EAST model, training-data preparation, training and evaluation scripts, and the QP22 checkpoint at 300,000 iterations.

## Repository structure

```text
EAST/
├── DCNedgev1.py                  # EAST model (class DCnv5)
├── common.py                    # Model components
├── create_lmdb_mfqev2.py         # Prepare training LMDBs from YUV videos
├── dataset/                     # LMDB training and YUV evaluation datasets
├── utils/                       # Original training and data utilities
├── train.py                     # Training
├── test.py                      # Y-PSNR evaluation
├── test_ssim.py                 # Y-SSIM evaluation
├── option_R3_mfqev2_4G.yml       # Training configuration
├── option_test_QP22.yml          # Released-checkpoint evaluation configuration
├── requirements.txt
└── exp/DCNedgev1_LDQP22_1920dataset_enlarge300x/
    ├── ckp_300000.pt
    ├── log.log
    └── log_test.log
```

The model retains its original filename `DCNedgev1.py` and class name `DCnv5` for compatibility with the checkpoint. It uses ordinary PyTorch operations and does not require a deformable-convolution extension.

## Installation

Use a Python environment with NVIDIA CUDA-enabled PyTorch for training and evaluation. Install PyTorch for your CUDA environment, then install the remaining dependencies:

```bash
git clone https://github.com/HUANGZiyin1/EAST.git
cd EAST
python -m pip install -r requirements.txt
```

Dependencies are PyTorch, NumPy, OpenCV, LMDB, PyYAML, tqdm, and scikit-image. Run the following commands from the repository root.

## Prepare the data

The original implementation uses the Y plane of **8-bit planar YUV444** videos. Keep the original filename convention, for example:

```text
Scene_1920x1080_30_8bit_300_444.yuv
```

Validation and test reconstructions use the same filename with a `rec` prefix:

```text
recScene_1920x1080_30_8bit_300_444.yuv
```

Prepare uncompressed ground-truth videos and their matching QP22 reconstructions:

```text
data/
├── train/
│   ├── raw/
│   └── QP22/
├── eval/
│   ├── raw/
│   └── QP22/
└── test/
    ├── raw/
    └── QP22/
```

Data locations are configured in the YAML files and the directory strings in `dataset/mfqev2.py`. The packaged root is `data/`. If using a different root, keep the YAML `dataset.train.root` and the dataset's root paths aligned.

### Build training LMDBs

```bash
python create_lmdb_mfqev2.py --opt_path option_R3_mfqev2_4G.yml
```

This creates:

```text
data/scc_LD_STDFgt22.lmdb/
data/scc_LD_STDFlq22.lmdb/
```

The original preparation code uses at most the first 300 frames of each video and divides them into non-overlapping seven-frame groups. Each group provides seven compressed input frames and one uncompressed center frame. Validation and testing read YUV files directly.

## Train

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --opt_path option_R3_mfqev2_4G.yml
```

The supplied configuration retains the original source settings:

| Setting | Value |
| --- | --- |
| Input frames | 7 |
| Crop size | 128 × 128 |
| Batch size per GPU | 16 |
| DataLoader workers per GPU | 6 |
| Dataset enlargement ratio | 300 |
| Random seed | 7 |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Training iterations | 300,000 |
| Validation/checkpoint interval | 5,000 iterations |

Training outputs are written to `exp/EAST_QP22_train/`. Set `train.exp_name` to a new experiment-directory name for a new run.

## Evaluate

The evaluation configuration selects the included checkpoint:

```text
exp/DCNedgev1_LDQP22_1920dataset_enlarge300x/ckp_300000.pt
```

Y-PSNR:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py --opt_path option_test_QP22.yml
```

Y-SSIM:

```bash
CUDA_VISIBLE_DEVICES=0 python test_ssim.py --opt_path option_test_QP22.yml
```

Both scripts average frame scores within each video, then average the video scores. `test.py` retains the original `log_test.log` output path and write behavior. `test_ssim.py` writes to `log_test_current.log` in the same experiment directory. The included `log.log` and `log_test.log` are the original experiment records.

## Implementation

The original model, training, LMDB preparation, dataset and utility logic are retained. Changes are limited to data/output paths and selecting `DCNedgev1` in `test.py`. The supplied checkpoint and historical logs are unchanged.

## Citation

If this work is useful in your research, please cite:

```bibtex
@article{huang2024east,
  title   = {Spatio-temporal feature learning for enhancing video quality based on screen content characteristics},
  author  = {Huang, Ziyin and Chan, Yui-Lam and Tsang, Sik-Ho and Kwong, Ngai-Wing and Lam, Kin-Man and Ling, Wing-Kuen},
  journal = {Journal of Visual Communication and Image Representation},
  volume  = {104},
  pages   = {104270},
  year    = {2024},
  doi     = {10.1016/j.jvcir.2024.104270},
  url     = {https://doi.org/10.1016/j.jvcir.2024.104270}
}
```

## Acknowledgements

The training and data-processing utilities build on [STDF-PyTorch](https://github.com/ryanxingql/stdf-pytorch). We thank its authors for sharing their code.
