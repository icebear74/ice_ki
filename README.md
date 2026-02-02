# ice_ki - Video Super Resolution Model (Option B)

## 🎯 Overview

This repository implements **Option B** comprehensive training improvements for the VSR+++ (Video Super Resolution Triple Plus) model. The model performs 3x upscaling of video frames using temporal information from 5 consecutive frames.

## ✨ Option B Improvements

### 1. **Mixed Precision Training**
- Utilizes PyTorch's Automatic Mixed Precision (AMP) with `autocast` and `GradScaler`
- Reduces VRAM usage from ~6GB to ~4.5GB
- Maintains training quality while improving speed

### 2. **Perceptual Loss**
- VGG19-based perceptual loss (features[:36])
- Combined with L1 loss: `Total = L1 + 0.1 * Perceptual`
- Improves visual quality and texture preservation

### 3. **Enhanced Model Architecture** (`model_vsrppp_v2.py`)
- **Adaptive Fusion**: Learnable 1x1 convolutions instead of naive addition
- **Learnable Temporal Weights**: Softmax-normalized weights for [backward, center, forward] frames
- **Layer Activity Tracking**: Real-time monitoring of each HeavyBlock's activation
- **Frame Stats**: Compatibility property for training interface

### 4. **Advanced UI with Live Metrics**
- **Loss Breakdown**: Separate display of L1, Perceptual, and Total losses
- **Layer Activity Trends**: Real-time trend calculation (⬆/⬇/═) for each layer
- **Top Layers Display**: Shows 5 most active layers with trend indicators
- **Cold Layers Warning**: Alerts about underutilized layers (<20% activity)
- **Convergence Status**: Visual indicator (Converging ✓, Plateauing ⚠, Diverging ✗)

### 5. **Enhanced TensorBoard Logging**
- Training metrics: L1, Perceptual, Total loss, Learning Rate
- Layer-specific activity: Individual tracking for all 30 blocks
- Activity distribution histogram
- Comprehensive validation metrics

### 6. **Data Augmentation**
- Horizontal flips (50% probability)
- Vertical flips (50% probability)
- Random rotations (90°, 180°, 270°)
- Applied only to training patches

### 7. **Learning Rate Scheduling**
- Cosine Annealing scheduler
- T_max = MAX_STEPS, eta_min = 1e-7
- Smooth learning rate decay for stable convergence

## 📊 Expected Performance

| Metric | Value |
|--------|-------|
| **PSNR** | ~31 dB |
| **Training Time** | ~5 days (100k steps) |
| **VRAM Usage** | ~4.5 GB |
| **Dataset** | 29,000 patches |
| **Batch Size** | 4 (effective: 12 with accumulation) |

## 🚀 Usage

### Installation

```bash
pip install -r requirements.txt
```

### Training

```bash
python train.py
```

**Interactive Controls:**
- `ENTER`: Open live setup menu
- `S`: Toggle layer sorting
- `P`: Pause/Resume training
- `V`: Instant validation

### Configuration

Edit parameters in the live menu or modify `train_config.json`:

```json
{
    "LR_EXPONENT": -4,
    "WEIGHT_DECAY": 0.01,
    "MAX_STEPS": 100000,
    "VAL_STEP_EVERY": 250,
    "SAVE_STEP_EVERY": 5000,
    "LOG_TBOARD_EVERY": 10,
    "ACCUMULATION_STEPS": 3,
    "SORT_BY_ACTIVITY": true
}
```

## 📁 Project Structure

```
ice_ki/
├── model_vsrppp_v2.py      # Enhanced VSR model
├── train.py                 # Training script with Option B improvements
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── /mnt/data/training/
    ├── checkpoints/         # Model checkpoints
    └── logs/                # TensorBoard logs
```

## 🔧 Model Architecture

```
Input: [B, 5, 3, H, W] (5 frames)
  ↓
Feature Extraction (3 → 96 channels)
  ↓
Adaptive Fusion
  ├─ Backward: F3 + F4 → Conv1x1 → 15 HeavyBlocks
  └─ Forward:  F0 + F1 → Conv1x1 → 15 HeavyBlocks
  ↓
Learnable Temporal Weighting
  weighted_sum = F2 * w[1] + backward * w[0] + forward * w[2]
  ↓
Fusion Conv3x3
  ↓
Upsampling (PixelShuffle 3x)
  ↓
Output: [B, 3, H*3, W*3] + Bilinear Anchor
```

## 📈 TensorBoard Monitoring

```bash
tensorboard --logdir /mnt/data/training/Universal/Mastermodell/Learn/logs/active_run
```

**Available Metrics:**
- `Training/Loss_L1`: L1 reconstruction loss
- `Training/Loss_Perceptual`: VGG perceptual loss
- `Training/Loss_Total`: Combined loss
- `Training/LearningRate`: Current learning rate
- `Layers/Block_XX`: Individual layer activities
- `Layers/ActivityDistribution`: Overall activity histogram

## 🎓 Fine-Tuning Preparation

After pretraining (100k steps), the model is ready for fine-tuning:

1. **Checkpoint**: Use `latest.pth` or milestone checkpoints
2. **Recommended Settings**:
   - Lower learning rate: `LR_EXPONENT = -5`
   - Smaller weight decay: `WEIGHT_DECAY = 0.001`
   - Targeted dataset: Replace with domain-specific data

## 🔍 Monitoring Training Health

**Good Signs:**
- Convergence status shows "Converging ✓"
- Most layers show 30-80% activity
- Loss trends downward smoothly
- Layer trends stable or slightly positive

**Warning Signs:**
- "Diverging ✗" status
- Many cold layers (<20% activity)
- Rapid negative trends in multiple layers
- Exploding loss values

## 📝 Checkpoint Format

```python
{
    'step': int,
    'epoch': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': dict,
    'scaler_state_dict': dict,
    'scheduler_state_dict': dict,
    'config': dict,
    'training_phase': str  # 'pretrain' or 'finetune'
}
```

## 🛡️ Key Improvements Over Base Model

1. **Stability**: Adaptive fusion prevents gradient explosion
2. **Efficiency**: Mixed precision reduces memory footprint
3. **Quality**: Perceptual loss improves visual fidelity
4. **Observability**: Real-time metrics for all 30 layers
5. **Convergence**: Learning rate scheduling ensures smooth training
6. **Robustness**: Data augmentation prevents overfitting

## 📚 References

- Model: VSR+++ (Video Super Resolution Triple Plus)
- Base Architecture: Residual learning with temporal propagation
- Perceptual Loss: VGG19 feature matching
- Training: Mixed Precision with gradient accumulation

