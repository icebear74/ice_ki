"""
VSR++ Manual Configuration

This file contains all important parameters for training.
Edit these values directly to configure your training run.
"""

# ============================================================================
# MODEL ARCHITECTURE PARAMETERS
# ============================================================================

# Number of feature channels (64, 96, 128, 160, 192, 256)
# Higher = more capacity but slower and more VRAM
N_FEATS = 128

# Total number of residual blocks (20, 24, 28, 32)
# Higher = more capacity but slower
N_BLOCKS = 32


# ============================================================================
# TRAINING BATCH PARAMETERS
# ============================================================================

# Batch size per iteration (1, 2, 3, 4, 6, 8)
# Higher = faster training but more VRAM
BATCH_SIZE = 4

# Gradient accumulation steps (1, 2, 3, 4)
# Effective batch = BATCH_SIZE * ACCUMULATION_STEPS
# Use accumulation if you need larger effective batch but have limited VRAM
ACCUMULATION_STEPS = 1


# ============================================================================
# LEARNING RATE PARAMETERS
# ============================================================================

# Initial learning rate as exponent (e.g., -5 means 1e-5 = 0.00001)
# Typical range: -6 to -4
LR_EXPONENT = -5

# Weight decay for AdamW optimizer (regularization)
# Typical range: 0.0001 to 0.01
WEIGHT_DECAY = 0.001

# Warmup steps (linear increase from 0 to max LR)
WARMUP_STEPS = 1000

# Maximum learning rate after warmup
MAX_LR = 1e-4

# Minimum learning rate at end of training
MIN_LR = 1e-6


# ============================================================================
# LOSS FUNCTION WEIGHTS
# ============================================================================

# Initial loss component weights (should sum to ~1.0)
L1_WEIGHT = 0.6      # L1 pixel loss
MS_WEIGHT = 0.2      # Multi-scale loss
GRAD_WEIGHT = 0.2    # Gradient loss


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

# Pin memory for faster GPU transfer (True/False)
PIN_MEMORY = True


# ============================================================================
# PATHS
# ============================================================================

# Training data root directory
DATA_ROOT = "/mnt/data/training/Universal/Mastermodell/Learn"

# Dataset root directory
DATASET_ROOT = "/mnt/data/training/Dataset/Universal/Mastermodell"


# ============================================================================
# ADAPTIVE SYSTEM (OPTIONAL)
# ============================================================================

# Enable adaptive loss weights (True/False)
# If True, loss weights will adjust automatically during training
ADAPTIVE_LOSS_WEIGHTS = True

# Enable adaptive gradient clipping (True/False)
# If True, gradient clip value will adjust automatically
ADAPTIVE_GRAD_CLIP = True

# Initial gradient clip value
INITIAL_GRAD_CLIP = 1.5


# ============================================================================
# HELPER FUNCTION
# ============================================================================

def get_config():
    """
    Returns configuration as a dictionary.
    This is used by the training system.
    """
    return {
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


def print_config():
    """Print current configuration in a readable format."""
    print("\n" + "="*80)
    print("CURRENT CONFIGURATION")
    print("="*80)
    
    print("\nMODEL ARCHITECTURE:")
    print(f"  Features (n_feats):     {N_FEATS}")
    print(f"  Blocks (n_blocks):      {N_BLOCKS}")
    
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
    
    print("\nLOSS WEIGHTS:")
    print(f"  L1 Weight:              {L1_WEIGHT}")
    print(f"  MS Weight:              {MS_WEIGHT}")
    print(f"  Grad Weight:            {GRAD_WEIGHT}")
    print(f"  Total:                  {L1_WEIGHT + MS_WEIGHT + GRAD_WEIGHT}")
    
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
    
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    # If run directly, print the configuration
    print_config()
