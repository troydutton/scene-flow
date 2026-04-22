# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements **ZeroFlow**, a scalable scene flow estimation system using knowledge distillation (ICLR 2024). The top-level contains two components:
- `zeroflow/`: Main scene flow codebase (primary focus)
- `OpenPCDet/`: LiDAR-based 3D object detection framework (supporting infrastructure)

> **Note**: The ZeroFlow codebase is considered deprecated; new development has moved to [SceneFlowZoo](https://github.com/kylevedder/SceneFlowZoo).

## Commands

All commands run from within `zeroflow/` (or inside the Docker container):

```bash
# Training
python train_pl.py <config_path> --gpus <num_gpus>
# Example: python train_pl.py configs/fastflow3d/argo/supervised.py --gpus 4

# Evaluation
python test_pl.py <config_path> <checkpoint_path> --gpus <num_gpus>

# Visualize predictions
python visualization/visualize_unsupervised_flow.py

# Generate paper figures/tables
python plot_performance.py

# AV2 competition submission format
python av2_scene_flow_competition_submit.py
```

### Docker (recommended for reproducible environments)
```bash
./launch.sh            # Main training container
./launch_waymo.sh      # Waymo preprocessing container
./launch_av2.sh        # AV2 challenge submission container
```

Docker containers expect datasets mounted at:
- Argoverse 2: `/efs/argoverse2/{train,val,test}`
- Waymo Open: `/efs/waymo_open_processed_flow/`

### Data Preprocessing
```bash
# Argoverse 2 LiDAR subsampling
python data_prep_scripts/argo/make_lidar_subset.py <source> <dest>

# Waymo (run inside Waymo Docker container)
python data_prep_scripts/waymo/rasterize_heightmap.py
python data_prep_scripts/waymo/extract_flow_and_remove_ground.py
```

## Architecture

### Data Pipeline
```
Raw sensor data (Argoverse 2 / Waymo)
    → SequenceLoader (dataloaders/)
    → SubsequenceFlowDataset (supervised or unsupervised)
    → PyTorch DataLoader
    → Model
    → Loss + PyTorch Lightning Trainer
```

### Configuration System
Configs live in `zeroflow/configs/` and use a `_base_` inheritance pattern. They are organized by method (`fastflow3d/`, `nsfp/`, `chodosh/`, `nearest_neighbor/`) and dataset (`argo/`, `waymo/`). Each config specifies model architecture, dataset paths, training hyperparameters, and loss type.

Base pseudo-image parameters are defined in `configs/pseudoimage.py`.

### Models (`zeroflow/models/`)
- **FastFlow3D** (`fast_flow_3d.py`): Primary model. Converts point clouds to 2D pseudo-images, runs a UNet encoder-decoder, then decodes to 3D flow vectors. Supports three loss modes: supervised, self-supervised, and distillation.
- **NSFP** (`nsfp.py`): Neural scene flow prior; used as the teacher for distillation.
- **JointFlow** (`joint_flow.py`): Joint 2D+3D flow estimation.
- **NearestNeighborFlow** / **Chodosh**: Baseline methods.

FastFlow3D sub-components:
- `models/embedders/`: Point cloud → pseudo-image (HardEmbedder, DynamicEmbedder)
- `models/backbones/`: UNet variants (FastFlowUNet, FastFlowUNetXL)
- `models/heads/`: Flow decoder

### Training Infrastructure
`model_wrapper.py` contains `ModelWrapper`, a PyTorch Lightning module that wraps all models, computes metrics (`EndpointDistanceMetricRawTorch`), and handles DDP training. TensorBoard logging is integrated via Lightning.

### Key Data Classes (`zeroflow/pointclouds/`)
- `PointCloud`: Core 3D point cloud representation
- `SE3` / `SE2`: Rigid body transforms used throughout the pipeline
- `losses/`: Warped point cloud loss functions for self-supervised training
