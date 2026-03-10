"""
VSR++ 7-Frame Configuration - Example Template

⚠️  IMPORTANT: Copy this file to config.py before using!

    cp config.py.example config.py

The config.py file is in .gitignore and will NOT be committed.
Edit config.py for your local setup.

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
N_BLOCKS = 28


# ============================================================================
# TRAINING BATCH PARAMETERS (Optimized for 7-Frame Model)
# ============================================================================

# Batch size per iteration
# Default: 2 (für 720_169 und 540 getestet; 720 bleibt bei batch=1 wegen OOM-Risiko)
BATCH_SIZE = 2

# Gradient accumulation steps
# Effektive Batch = BATCH_SIZE * ACCUMULATION_STEPS = 2 * 4 = 8 (für 720_169)
# Neuer Standard passt zu 720_169-Konfiguration
ACCUMULATION_STEPS = 4


# ============================================================================
# ADAPTIVE BATCH CONFIGURATION (Per-Size Optimized)
# ============================================================================
# Basierend auf gemessenen VRAM-Werten aus config_test_results.txt
# (7f | 26b | 72f | FP32 - aktive Modellkonfiguration)
#
# 720_169 (720×405) - Vollbilder 16:9:
#   BS=2, A=4 → eff. Batch=8 | VRAM: ~5.14 GB ✅
#
# 540 (540×540) - Crops aus 1080p:
#   BS=2, A=3 → eff. Batch=6 | VRAM: ~5.15 GB ✅
#
# 720 (720×720) - 4K Crops (VRAM-kritisch!):
#   BS=1, A=4 → eff. Batch=4 | VRAM: ~6.14 GB ✅  (BS=2 = OOM!)

ADAPTIVE_BATCH_CONFIG = {
    '720_169': {'batch': 2, 'accum': 4},   # eff=8 | ~5.14 GB | Vollbilder 16:9
    '540':     {'batch': 2, 'accum': 3},   # eff=6 | ~5.15 GB | 1080p Crops
    '720':     {'batch': 1, 'accum': 4},   # eff=4 | ~6.14 GB | 4K Crops (BS=1 pflicht!)
}


# ============================================================================
# LEARNING RATE PARAMETERS
# ============================================================================

# Initial learning rate as exponent (e.g., -5 means 1e-5 = 0.00001)
LR_EXPONENT = -5

# Weight decay for AdamW optimizer (regularization)
WEIGHT_DECAY = 5e-4

# Warmup steps (linear increase from 0 to max LR)
WARMUP_STEPS = 2000

# Maximum learning rate after warmup
MAX_LR = 1.5e-4

# Minimum learning rate at end of training
# Increased from 1e-7 to 1e-5 so LR stays usable longer
MIN_LR = 1e-5


# ============================================================================
# LOSS FUNCTION WEIGHTS (Optimized for VGG Perceptual Loss)
# ============================================================================

# L1 pixel loss - PRIMARY loss component
L1_WEIGHT = 0.60

# Multi-scale loss - DISABLED (redundant with perceptual)
MS_WEIGHT = 0.20

# Gradient loss - DISABLED (redundant with perceptual)
GRAD_WEIGHT = 0.20

# VGG-based perceptual loss - ENABLED for sharpness feedback
# This uses pretrained VGG16 weights, providing REAL perceptual guidance
# (unlike the previous untrained custom loss that caused stagnation)
PERCEPTUAL_WEIGHT = 0.00


# ============================================================================
# TRAINING SCHEDULE
# ============================================================================

# Maximum training steps
# Increased from 100000 to 150000 to prevent LR from dropping too fast
MAX_STEPS = 150000

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
NUM_WORKERS = 5

# Pin memory for faster GPU transfer
PIN_MEMORY = True


# ============================================================================
# PATHS (Match dataset_generator_v2 output structure)
# ============================================================================

# Dataset root directory - base directory for all datasets
# This matches runtime_config.json "data.root"
DATASET_ROOT = "/mnt/data/training/datasetNeu"

# Default dataset name (category) - used if runtime_config.json not found
# Options: 'master', 'universal', 'space', 'toon' (lowercase)
DEFAULT_DATASET_NAME = "master"

# For backward compatibility - will be overridden by runtime_config.json
# New structure: DATASET_ROOT/dataset_name/patches/{size_key}/GT
# Old structure (deprecated): DATASET_ROOT/Master/MasterModel/Learn/Patches/GT
DATA_ROOT = f"{DATASET_ROOT}/{DEFAULT_DATASET_NAME}"


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
# NOTE: Tesla P4 has NO hardware FP16 support (only emulated).
# AMP adds ~3GB overhead without providing any speedup on this GPU.
# Disabled to save VRAM and improve training speed.
USE_AMP = False


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
        'ADAPTIVE_BATCH_CONFIG': ADAPTIVE_BATCH_CONFIG,
        
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
    
    # Add AMP setting (always include so training code can rely on its presence)
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
    
    print("\nBATCH SETTINGS (Per-Size Optimiert):")
    print(f"  Standard Batch Size:    {BATCH_SIZE}")
    print(f"  Standard Accum Steps:   {ACCUMULATION_STEPS}")
    print(f"  Effektive Batch Size:   {BATCH_SIZE * ACCUMULATION_STEPS}")
    print(f"  Per-Size Konfiguration (gemessene VRAM-Werte):")
    for size_key, cfg in ADAPTIVE_BATCH_CONFIG.items():
        eff = cfg['batch'] * cfg['accum']
        print(f"    {size_key:<10}: BS={cfg['batch']}, A={cfg['accum']} → eff={eff}")
    
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
    print(f"  Dataset Root:           {DATASET_ROOT}")
    print(f"  Category (dataset_name): {DEFAULT_DATASET_NAME}")
    
    # Try to load runtime_config.json to show actual configuration
    import os
    import json
    runtime_config_file = os.path.join(os.path.dirname(__file__), "runtime_config.json")
    
    if os.path.exists(runtime_config_file):
        try:
            with open(runtime_config_file, 'r') as f:
                rt_config = json.load(f)
            dataset_root = rt_config.get('data', {}).get('root', DATASET_ROOT)
            dataset_name = rt_config.get('data', {}).get('dataset_name', DEFAULT_DATASET_NAME)
            
            print(f"\n  ✓ runtime_config.json found:")
            print(f"    Root:                 {dataset_root}")
            print(f"    Dataset Name:         {dataset_name}")
            
            # Show expected structure for each size_key
            size_dist = rt_config.get('size_distribution', {})
            enabled_sizes = [k for k, v in size_dist.items() if v > 0]
            
            if enabled_sizes:
                print(f"\n  Expected Structure (NEW - size-specific):")
                for size_key in enabled_sizes:
                    print(f"    Training {size_key}:")
                    print(f"      {dataset_root}/{dataset_name}/patches/{size_key}/GT/")
                    print(f"      {dataset_root}/{dataset_name}/patches/{size_key}/LR_7frames/")
                
                # Show validation structure
                val_sizes = rt_config.get('validation', {}).get('sizes', enabled_sizes)
                if val_sizes:
                    print(f"\n    Validation:")
                    for size_key in val_sizes:
                        print(f"      {dataset_root}/{dataset_name}/val/{size_key}/GT/")
                        print(f"      (LR auto-found in patches/{size_key}/LR_7frames/)")
        except Exception as e:
            print(f"\n  ⚠ Could not parse runtime_config.json: {e}")
            print(f"  Using default paths (backward compatible)")
    else:
        print(f"\n  ⚠ runtime_config.json not found")
        print(f"  Expected at: {runtime_config_file}")
        print(f"  Using default single-size structure:")
        print(f"    {DATA_ROOT}/patches/540/GT/")
        print(f"    {DATA_ROOT}/patches/540/LR_7frames/")
    
    print("\nADAPTIVE SYSTEM:")
    print(f"  Adaptive Loss Weights:  {ADAPTIVE_LOSS_WEIGHTS}")
    print(f"  Adaptive Grad Clip:     {ADAPTIVE_GRAD_CLIP}")
    print(f"  Initial Grad Clip:      {INITIAL_GRAD_CLIP}")
    
    print("\nPERFORMANCE:")
    print(f"  Mixed Precision (AMP):  {USE_AMP}")
    
    print("\n" + "="*80)
    print("CONFIGURATION NOTES:")
    print("  - 7-frame model with 72 features and 26 blocks")
    print("  - Dataset structure matches dataset_generator_v2 output")
    print("  - Lowercase category names (master, universal, space, toon)")
    print("  - Size-specific directories: patches/{size_key}/ and val/{size_key}/")
    print("  - Validation LR files auto-found in patches/{size_key}/LR_7frames/")
    print("  - VGG perceptual loss enabled for quality")
    print("  - Gradient accumulation für per-size optimierte effektive Batch-Größen")
    print("="*80 + "\n")


if __name__ == '__main__':
    # If run directly, print the configuration
    print_config()
