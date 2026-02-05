"""
VSR++ Tesla P4 Optimized Configuration

This configuration is specifically optimized for Tesla P4 hardware (8GB VRAM).
Key optimizations:
- Reduced model size (64 features instead of 128)
- Reduced depth (24 blocks instead of 36)
- Gradient accumulation for effective batch size 16
- VGG-based perceptual loss enabled for better sharpness
"""

# ============================================================================
# MODEL ARCHITECTURE PARAMETERS (Optimized for Tesla P4)
# ============================================================================

# Number of feature channels - REDUCED for Tesla P4
# Was: 128 (too heavy for P4)
# Now: 64 (optimal for 8GB VRAM)
N_FEATS = 64

# Total number of residual blocks - REDUCED for Tesla P4
# Was: 36 (too deep)
# Now: 24 (balanced capacity/speed)
N_BLOCKS = 24


# ============================================================================
# TRAINING BATCH PARAMETERS (Optimized for Tesla P4)
# ============================================================================

# Batch size per iteration
# Keep at 4 to fit in VRAM
BATCH_SIZE = 4

# Gradient accumulation steps
# Effective batch = BATCH_SIZE * ACCUMULATION_STEPS = 4 * 4 = 16
# This provides stable gradients without excessive VRAM usage
ACCUMULATION_STEPS = 4


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
MAX_LR = 1e-4

# Minimum learning rate at end of training
MIN_LR = 1e-6


# ============================================================================
# LOSS FUNCTION WEIGHTS (Optimized for VGG Perceptual Loss)
# ============================================================================

# L1 pixel loss - PRIMARY loss component
L1_WEIGHT = 1.0

# Multi-scale loss - DISABLED (redundant with perceptual)
MS_WEIGHT = 0.0

# Gradient loss - DISABLED (redundant with perceptual)
GRAD_WEIGHT = 0.0

# VGG-based perceptual loss - ENABLED for sharpness feedback
# This uses pretrained VGG16 weights, providing REAL perceptual guidance
# (unlike the previous untrained custom loss that caused stagnation)
PERCEPTUAL_WEIGHT = 0.1


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
# PATHS
# ============================================================================

# Training data root directory
DATA_ROOT = "/mnt/data/training/Universal/Mastermodell/Learn"

# Dataset root directory
DATASET_ROOT = "/mnt/data/training/Dataset/Universal/Mastermodell"


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
    print("TESLA P4 OPTIMIZED CONFIGURATION")
    print("="*80)
    
    print("\nMODEL ARCHITECTURE (P4 Optimized):")
    print(f"  Features (n_feats):     {N_FEATS} (reduced from 128)")
    print(f"  Blocks (n_blocks):      {N_BLOCKS} (reduced from 36)")
    
    print("\nBATCH SETTINGS:")
    print(f"  Batch Size:             {BATCH_SIZE}")
    print(f"  Accumulation Steps:     {ACCUMULATION_STEPS}")
    print(f"  Effective Batch Size:   {BATCH_SIZE * ACCUMULATION_STEPS}")
    
    print("\nLEARNING RATE:")
    print(f"  Initial LR:             {10**LR_EXPONENT:.2e} (10^{LR_EXPONENT})")
    print(f"  Max LR:                 {MAX_LR:.2e}")
    print(f"  Min LR:                 {MIN_LR:.2e}")
    print(f"  Weight Decay:           {WEIGHT_DECAY}")
    print(f"  Warmup Steps:           {WARMUP_STEPS:,}")
    
    print("\nLOSS WEIGHTS (VGG Perceptual Enabled):")
    print(f"  L1 Weight:              {L1_WEIGHT} (primary)")
    print(f"  MS Weight:              {MS_WEIGHT} (disabled)")
    print(f"  Grad Weight:            {GRAD_WEIGHT} (disabled)")
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
    
    print("\nADAPTIVE SYSTEM:")
    print(f"  Adaptive Loss Weights:  {ADAPTIVE_LOSS_WEIGHTS}")
    print(f"  Adaptive Grad Clip:     {ADAPTIVE_GRAD_CLIP}")
    print(f"  Initial Grad Clip:      {INITIAL_GRAD_CLIP}")
    
    print("\nPERFORMANCE:")
    print(f"  Mixed Precision (AMP):  {USE_AMP}")
    
    print("\n" + "="*80)
    print("OPTIMIZATION NOTES:")
    print("  - Model size reduced to fit Tesla P4's 8GB VRAM")
    print("  - VGG perceptual loss enabled to fix training stagnation")
    print("  - Gradient accumulation used for effective batch size 16")
    print("  - AMP enabled for faster training on Tesla P4")
    print("="*80 + "\n")


if __name__ == '__main__':
    # If run directly, print the configuration
    print_config()
