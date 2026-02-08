"""
VSR++ 7-Frame Configuration - Optimized for Tesla P4

This configuration is specifically optimized for 7-frame VSR training on Tesla P4 hardware (8GB VRAM).
Key parameters:
- 72 feature channels (optimized for 7-frame model)
- 26 residual blocks (balanced depth for quality)
- Gradient accumulation for effective batch size
- Matches dataset_generator_v2 output structure
"""

# ============================================================================
# MODEL ARCHITECTURE PARAMETERS (7-Frame Optimized)
# ============================================================================

# Number of feature channels - Optimized for 7-frame model
# 72 features provides good capacity while staying within VRAM limits
N_FEATS = 72

# Total number of residual blocks - Optimized for quality
# 26 blocks provides excellent capacity for 7-frame processing
N_BLOCKS = 26


# ============================================================================
# TRAINING BATCH PARAMETERS (Optimized for 7-Frame Model)
# ============================================================================

# Batch size per iteration
# Keep at 1 for safety with 7-frame model (VRAM tested: ~3.77 GB @ batch=1)
BATCH_SIZE = 1

# Gradient accumulation steps
# Effective batch = BATCH_SIZE * ACCUMULATION_STEPS = 1 * 6 = 6
# This provides stable gradients without excessive VRAM usage
ACCUMULATION_STEPS = 6


# ============================================================================
# LEARNING RATE PARAMETERS
# ============================================================================

# Initial learning rate as exponent (e.g., -5 means 1e-5 = 0.00001)
LR_EXPONENT = -5

# Weight decay for AdamW optimizer (regularization)
WEIGHT_DECAY = 0.001

# Warmup steps (linear increase from 0 to max LR)
WARMUP_STEPS = 1000

# Maximum learning rate after warmup
MAX_LR = 1.5e-4

# Minimum learning rate at end of training
MIN_LR = 1e-7


# ============================================================================
# LOSS FUNCTION WEIGHTS (Optimized for VGG Perceptual Loss)
# ============================================================================

# L1 pixel loss - PRIMARY loss component
L1_WEIGHT = 0.55

# Multi-scale loss - DISABLED (redundant with perceptual)
MS_WEIGHT = 0.20

# Gradient loss - DISABLED (redundant with perceptual)
GRAD_WEIGHT = 0.20

# VGG-based perceptual loss - ENABLED for sharpness feedback
# This uses pretrained VGG16 weights, providing REAL perceptual guidance
# (unlike the previous untrained custom loss that caused stagnation)
PERCEPTUAL_WEIGHT = 0.05


# ============================================================================
# TRAINING SCHEDULE
# ============================================================================

# Maximum training steps
MAX_STEPS = 100000

# Validation frequency (run validation every N steps)
VAL_STEP_EVERY = 500

# Regular checkpoint saving frequency (every N steps)
SAVE_STEP_EVERY = 10000

# TensorBoard logging frequency (every N steps)
LOG_TBOARD_EVERY = 100

# Histogram logging frequency (every N steps)
HIST_STEP_EVERY = 500


# ============================================================================
# DATA LOADING
# ============================================================================

# Number of worker threads for data loading
NUM_WORKERS = 4

# Pin memory for faster GPU transfer
PIN_MEMORY = True


# ============================================================================
# PATHS (Match dataset_generator_v2 output structure)
# ============================================================================

# Training data root directory - matches generator_config.json output_base_dir
# Generator creates: datasetNeu/Master/MasterModel/Learn/Patches/GT, Patches/LR, etc.
# VSRDataset expects: dataset_root/Patches/GT and dataset_root/Patches/LR
DATA_ROOT = "/mnt/data/training/datasetNeu/Master/MasterModel/Learn"

# Dataset root directory (for checkpoints and logs)
DATASET_ROOT = "/mnt/data/training/datasetNeu"


# ============================================================================
# ADAPTIVE SYSTEM
# ============================================================================

# Enable adaptive loss weights
ADAPTIVE_LOSS_WEIGHTS = True

# Enable adaptive gradient clipping
ADAPTIVE_GRAD_CLIP = True

# Initial gradient clip value
INITIAL_GRAD_CLIP = 1.5


# ============================================================================
# MIXED PRECISION TRAINING (AMP)
# ============================================================================

# Enable Automatic Mixed Precision for faster training on Tesla P4
# Tesla P4 supports FP16, which can significantly speed up training
USE_AMP = True


# ============================================================================
# HELPER FUNCTION
# ============================================================================

def get_config():
    """
    Returns configuration as a dictionary.
    This is used by the training system.
    """
    config = {
        # Model
        'N_FEATS': N_FEATS,
        'N_BLOCKS': N_BLOCKS,
        
        # Batch
        'BATCH_SIZE': BATCH_SIZE,
        'ACCUMULATION_STEPS': ACCUMULATION_STEPS,
        
        # Learning rate
        'LR_EXPONENT': LR_EXPONENT,
        'WEIGHT_DECAY': WEIGHT_DECAY,
        'WARMUP_STEPS': WARMUP_STEPS,
        'MAX_LR': MAX_LR,
        'MIN_LR': MIN_LR,
        
        # Loss weights
        'L1_WEIGHT': L1_WEIGHT,
        'MS_WEIGHT': MS_WEIGHT,
        'GRAD_WEIGHT': GRAD_WEIGHT,
        'PERCEPTUAL_WEIGHT': PERCEPTUAL_WEIGHT,
        
        # Training schedule
        'MAX_STEPS': MAX_STEPS,
        'VAL_STEP_EVERY': VAL_STEP_EVERY,
        'SAVE_STEP_EVERY': SAVE_STEP_EVERY,
        'LOG_TBOARD_EVERY': LOG_TBOARD_EVERY,
        'HIST_STEP_EVERY': HIST_STEP_EVERY,
        
        # Data loading
        'NUM_WORKERS': NUM_WORKERS,
        'PIN_MEMORY': PIN_MEMORY,
        
        # Paths
        'DATA_ROOT': DATA_ROOT,
        'DATASET_ROOT': DATASET_ROOT,
        
        # Adaptive system
        'ADAPTIVE_LOSS_WEIGHTS': ADAPTIVE_LOSS_WEIGHTS,
        'ADAPTIVE_GRAD_CLIP': ADAPTIVE_GRAD_CLIP,
        'INITIAL_GRAD_CLIP': INITIAL_GRAD_CLIP,
    }
    
    # Add AMP if enabled
    if USE_AMP:
        config['USE_AMP'] = USE_AMP
    
    return config


def print_config():
    """Print current configuration in a readable format."""
    print("\n" + "="*80)
    print("7-FRAME VSR CONFIGURATION (Tesla P4 Optimized)")
    print("="*80)
    
    print("\nMODEL ARCHITECTURE (7-Frame Optimized):")
    print(f"  Features (n_feats):     {N_FEATS}")
    print(f"  Blocks (n_blocks):      {N_BLOCKS}")
    
    print("\nBATCH SETTINGS (VRAM-Safe):")
    print(f"  Batch Size:             {BATCH_SIZE}")
    print(f"  Accumulation Steps:     {ACCUMULATION_STEPS}")
    print(f"  Effective Batch Size:   {BATCH_SIZE * ACCUMULATION_STEPS}")
    print(f"  Estimated VRAM:         ~3.77 GB @ batch=1")
    
    print("\nLEARNING RATE:")
    print(f"  Initial LR:             {10**LR_EXPONENT:.2e} (10^{LR_EXPONENT})")
    print(f"  Max LR:                 {MAX_LR:.2e}")
    print(f"  Min LR:                 {MIN_LR:.2e}")
    print(f"  Weight Decay:           {WEIGHT_DECAY}")
    print(f"  Warmup Steps:           {WARMUP_STEPS:,}")
    
    print("\nLOSS WEIGHTS (VGG Perceptual Enabled):")
    print(f"  L1 Weight:              {L1_WEIGHT} (primary)")
    print(f"  MS Weight:              {MS_WEIGHT}")
    print(f"  Grad Weight:            {GRAD_WEIGHT}")
    print(f"  Perceptual Weight:      {PERCEPTUAL_WEIGHT} (VGG16-based)")
    print(f"  Total:                  {L1_WEIGHT + MS_WEIGHT + GRAD_WEIGHT + PERCEPTUAL_WEIGHT}")
    
    print("\nTRAINING SCHEDULE:")
    print(f"  Max Steps:              {MAX_STEPS:,}")
    print(f"  Validation Every:       {VAL_STEP_EVERY:,} steps")
    print(f"  Save Checkpoint Every:  {SAVE_STEP_EVERY:,} steps")
    print(f"  TensorBoard Log Every:  {LOG_TBOARD_EVERY:,} steps")
    
    print("\nDATA LOADING:")
    print(f"  Workers:                {NUM_WORKERS}")
    print(f"  Pin Memory:             {PIN_MEMORY}")
    
    print("\nDATASET PATHS:")
    print(f"  Data Root:              {DATA_ROOT}")
    print(f"  Dataset Root:           {DATASET_ROOT}")
    print(f"  Expected Structure:")
    print(f"    {DATA_ROOT}/Patches/GT/")
    print(f"    {DATA_ROOT}/Patches/LR/")
    print(f"    {DATA_ROOT}/Val/GT/")
    
    print("\nADAPTIVE SYSTEM:")
    print(f"  Adaptive Loss Weights:  {ADAPTIVE_LOSS_WEIGHTS}")
    print(f"  Adaptive Grad Clip:     {ADAPTIVE_GRAD_CLIP}")
    print(f"  Initial Grad Clip:      {INITIAL_GRAD_CLIP}")
    
    print("\nPERFORMANCE:")
    print(f"  Mixed Precision (AMP):  {USE_AMP}")
    
    print("\n" + "="*80)
    print("CONFIGURATION NOTES:")
    print("  - 7-frame model with 72 features and 26 blocks")
    print("  - Batch=1 for VRAM safety (tested at ~3.77 GB)")
    print("  - Dataset paths match dataset_generator_v2 output")
    print("  - Generator creates: Master/MasterModel/Learn/Patches/...")
    print("  - VGG perceptual loss enabled for quality")
    print("  - Gradient accumulation for effective batch size 6")
    print("="*80 + "\n")


if __name__ == '__main__':
    # If run directly, print the configuration
    print_config()
