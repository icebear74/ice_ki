# VSR+++ Option B - Testing & Validation Guide

## 🧪 Pre-Training Checklist

Before starting training, verify all components are working correctly.

### 1. Environment Setup Test

```bash
# Run setup script
./setup_env.sh

# Expected output:
# - ✓ Python version check passed
# - ✓ Virtual environment created
# - ✓ PyTorch installed with CUDA
# - ✓ All dependencies installed
# - ✓ Model import successful
```

### 2. Import Test

```bash
source venv/bin/activate

# Test all imports
python << 'EOF'
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torchvision.models import vgg19
from model_vsrppp_v2 import VSRTriplePlus_3x
import cv2
import numpy as np

print("✓ All imports successful")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
EOF
```

### 3. Model Instantiation Test

```bash
python << 'EOF'
import torch
from model_vsrppp_v2 import VSRTriplePlus_3x

# Create model
model = VSRTriplePlus_3x(n_blocks=30)
print("✓ Model created")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")

# Test get_layer_activity
activities = model.get_layer_activity()
print(f"✓ get_layer_activity() returns {len(activities)} values")
assert len(activities) == 30, "Should have 30 layer activities"

# Test frame_stats
stats = model.frame_stats
print(f"✓ frame_stats: {stats}")
assert isinstance(stats, dict), "frame_stats should be a dict"

# Test forward pass
dummy_input = torch.randn(1, 5, 3, 64, 64)
output = model(dummy_input)
print(f"✓ Forward pass: {dummy_input.shape} -> {output.shape}")
assert output.shape == (1, 3, 192, 192), f"Output shape should be (1, 3, 192, 192), got {output.shape}"

print("\n✅ All model tests passed!")
EOF
```

### 4. Mixed Precision Test

```bash
python << 'EOF'
import torch
from torch.cuda.amp import autocast, GradScaler
from model_vsrppp_v2 import VSRTriplePlus_3x

if torch.cuda.is_available():
    device = torch.device('cuda')
    model = VSRTriplePlus_3x(n_blocks=30).to(device)
    scaler = GradScaler()
    
    # Test mixed precision forward pass
    dummy_input = torch.randn(2, 5, 3, 64, 64).to(device)
    dummy_target = torch.randn(2, 3, 192, 192).to(device)
    
    with autocast():
        output = model(dummy_input)
        loss = torch.nn.functional.l1_loss(output, dummy_target)
    
    scaler.scale(loss).backward()
    scaler.step(torch.optim.Adam(model.parameters()))
    scaler.update()
    
    print("✓ Mixed precision training works")
    print(f"Loss: {loss.item():.6f}")
    print("\n✅ Mixed precision test passed!")
else:
    print("⚠ CUDA not available, skipping GPU test")
EOF
```

### 5. VGG Perceptual Loss Test

```bash
python << 'EOF'
import torch
import torch.nn as nn
from torchvision.models import vgg19

class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        if torch.cuda.is_available():
            vgg = vgg19(pretrained=True).features[:36].eval().cuda()
        else:
            vgg = vgg19(pretrained=True).features[:36].eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.criterion = nn.L1Loss()
    
    def forward(self, x, y):
        x_vgg = self.vgg(x)
        y_vgg = self.vgg(y)
        return self.criterion(x_vgg, y_vgg)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

criterion = VGGLoss()
x = torch.randn(1, 3, 192, 192).to(device)
y = torch.randn(1, 3, 192, 192).to(device)

loss = criterion(x, y)
print(f"✓ VGG Perceptual Loss: {loss.item():.6f}")
print("\n✅ Perceptual loss test passed!")
EOF
```

### 6. Data Augmentation Test

```bash
python << 'EOF'
import numpy as np
import random
import torch

# Simulate data augmentation
def test_augmentation():
    # Create dummy data
    lr_frames = [np.random.rand(180, 180, 3) for _ in range(5)]
    gt = np.random.rand(540, 540, 3)
    
    # Horizontal flip
    if random.random() > 0.5:
        lr_frames = [f[:, ::-1].copy() for f in lr_frames]
        gt = gt[:, ::-1].copy()
    
    # Vertical flip
    if random.random() > 0.5:
        lr_frames = [f[::-1, :].copy() for f in lr_frames]
        gt = gt[::-1, :].copy()
    
    # Random rotation
    k = random.choice([0, 1, 2, 3])
    if k > 0:
        lr_frames = [np.rot90(f, k).copy() for f in lr_frames]
        gt = np.rot90(gt, k).copy()
    
    return lr_frames, gt

# Test multiple times
for i in range(10):
    lrs, gt = test_augmentation()
    assert all(lr.shape == (180, 180, 3) for lr in lrs), "LR frames shape mismatch"
    assert gt.shape == (540, 540, 3), "GT shape mismatch"

print("✓ Data augmentation works correctly")
print("\n✅ Augmentation test passed!")
EOF
```

### 7. Checkpoint Save/Load Test

```bash
python << 'EOF'
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler
from model_vsrppp_v2 import VSRTriplePlus_3x
import tempfile
import os

# Create model and optimizer
model = VSRTriplePlus_3x(n_blocks=30)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
scaler = GradScaler()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

# Save checkpoint
with tempfile.TemporaryDirectory() as tmpdir:
    checkpoint = {
        'step': 100,
        'epoch': 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': {'LR_EXPONENT': -4},
        'training_phase': 'pretrain'
    }
    
    ckpt_path = os.path.join(tmpdir, 'test.pth')
    torch.save(checkpoint, ckpt_path)
    print(f"✓ Checkpoint saved: {os.path.getsize(ckpt_path) / 1024 / 1024:.2f} MB")
    
    # Load checkpoint
    loaded = torch.load(ckpt_path)
    model.load_state_dict(loaded['model_state_dict'])
    optimizer.load_state_dict(loaded['optimizer_state_dict'])
    scaler.load_state_dict(loaded['scaler_state_dict'])
    scheduler.load_state_dict(loaded['scheduler_state_dict'])
    
    print("✓ Checkpoint loaded successfully")
    print(f"  Step: {loaded['step']}")
    print(f"  Epoch: {loaded['epoch']}")
    print(f"  Phase: {loaded['training_phase']}")

print("\n✅ Checkpoint test passed!")
EOF
```

## 🎯 Training Validation

### Quick Training Test (1 iteration)

```bash
# This will NOT actually train but verifies the training loop works
# You'll need to create minimal test data first

python << 'EOF'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torchvision.models import vgg19
from model_vsrppp_v2 import VSRTriplePlus_3x

print("Setting up training components...")

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VSRTriplePlus_3x(n_blocks=30).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
scaler = GradScaler()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

# Loss functions
l1_criterion = nn.L1Loss()

class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(pretrained=True).features[:36].eval().to(device)
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.criterion = nn.L1Loss()
    
    def forward(self, x, y):
        x_vgg = self.vgg(x)
        y_vgg = self.vgg(y)
        return self.criterion(x_vgg, y_vgg)

perceptual_criterion = VGGLoss()

print("Running test iteration...")

# Dummy batch
lrs = torch.randn(2, 5, 3, 64, 64).to(device)
gt = torch.randn(2, 3, 192, 192).to(device)

# Training step
model.train()
optimizer.zero_grad()

with autocast():
    output = model(lrs)
    loss_l1 = l1_criterion(output, gt)
    loss_perc = perceptual_criterion(output, gt)
    loss = loss_l1 + 0.1 * loss_perc

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
scheduler.step()

print(f"✓ Training iteration completed")
print(f"  L1 Loss: {loss_l1.item():.6f}")
print(f"  Perceptual Loss: {loss_perc.item():.6f}")
print(f"  Total Loss: {loss.item():.6f}")
print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.2e}")

# Test layer activity
activities = model.get_layer_activity()
print(f"✓ Layer activities: {len(activities)} layers")
print(f"  Max activity: {max(activities):.6f}")
print(f"  Min activity: {min(activities):.6f}")

print("\n✅ Training loop test passed!")
EOF
```

## 📊 Expected Results

### System Requirements
- **Python**: 3.8+
- **CUDA**: 11.8+ (recommended)
- **RAM**: 16GB+ 
- **VRAM**: 6GB+ (4.5GB with mixed precision)
- **Storage**: 50GB+ for checkpoints and logs

### Performance Metrics
- **Training Speed**: ~2-3 seconds/iteration (with CUDA)
- **VRAM Usage**: ~4.5GB with mixed precision
- **Model Size**: ~3.5M parameters
- **Checkpoint Size**: ~40-50MB per checkpoint

### Training Health Indicators
- Loss should decrease over time
- Layer activities should stabilize after ~1000 steps
- No NaN or Inf values in losses
- Most layers should show 30-80% activity

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
- Reduce BATCH_SIZE in train.py (default: 4)
- Increase ACCUMULATION_STEPS
- Ensure other GPU processes are closed

**2. VGG Model Download Issues**
- VGG19 will auto-download on first run
- Requires internet connection
- Downloads to `~/.cache/torch/hub/checkpoints/`

**3. Dataset Not Found**
- Verify dataset paths in train.py
- Ensure directory structure is correct
- Check read permissions

**4. Training Crashes**
- Check VRAM usage
- Verify CUDA compatibility
- Update PyTorch to latest version

## ✅ Final Checklist

Before starting full training:

- [ ] Environment setup completed (`./setup_env.sh`)
- [ ] All import tests passed
- [ ] Model instantiation works
- [ ] Mixed precision works (if using CUDA)
- [ ] VGG perceptual loss works
- [ ] Data augmentation verified
- [ ] Checkpoint save/load works
- [ ] Dataset paths configured correctly
- [ ] TensorBoard accessible
- [ ] Sufficient disk space available

## 🎓 Next Steps

Once all tests pass:

1. **Configure paths** in `train.py` if needed
2. **Start training**: `python train.py`
3. **Monitor with TensorBoard**: `tensorboard --logdir /path/to/logs`
4. **Review README.md** for training tips and controls

Good luck with your training! 🚀
