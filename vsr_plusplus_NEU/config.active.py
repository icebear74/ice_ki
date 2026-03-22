"""
VSR++ 7-Frame Configuration - Example Template

⚠️  IMPORTANT: Copy this file to config.py before using!

    cp config.py.example config.py

The config.py file is in .gitignore and will NOT be committed.
Edit config.py for your local setup.

This configuration is specifically optimized for 7-frame VSR training on Tesla P100 hardware (16GB VRAM).
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
# ADAPTIVE BATCH CONFIGURATION (Per-Size — einzige Wahrheitsquelle!)
# ============================================================================
# Diese Werte werden DIREKT im Training verwendet — kein dynamisches Ermitteln,
# keine Runtime-Überschreibung.  Basierend auf gemessenen VRAM-Werten
# (7f | 28b | 72f | AMP+FP16 - aktive Modellkonfiguration, Tesla P100 16GB).
#
# 720_169 (720×405) - Vollbilder 16:9:
#   BS=4, A=2 → eff. Batch=8
#
# 540 (540×540) - Crops aus 1080p:
#   BS=4, A=2 → eff. Batch=8
#
# 720 (720×720) - 4K Crops:
#   BS=4, A=2 → eff. Batch=8
#
# WICHTIG: Für jede neue size_key hier einen Eintrag anlegen!
# Training bricht mit klarem Fehler ab, wenn ein size_key fehlt.

ADAPTIVE_BATCH_CONFIG = {
    '720_169': {'batch': 8, 'accum': 1},   # eff=8 | ~5.14 GB | Vollbilder 16:9
    '540':     {'batch': 8, 'accum': 2},   # eff=6 | ~5.15 GB | 1080p Crops
    '720':     {'batch': 6, 'accum': 1},   # eff=4 | ~6.14 GB | 4K Crops (BS=1 pflicht!)
}


# ============================================================================
# TRAINING BATCH PARAMETERS (Fallback für single-size Modus)
# ============================================================================
# Diese Werte werden NUR im single-size Fallback-Pfad genutzt.
# Im Multi-Size-Training (Normalfall) gelten ausschließlich ADAPTIVE_BATCH_CONFIG.

# Batch size per iteration (single-size fallback — entspricht 720_169)
BATCH_SIZE = 2

# Gradient accumulation steps (single-size fallback — entspricht 720_169)
ACCUMULATION_STEPS = 4


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
# DATA LOADING  –  2-Stage Prefetch Pipeline
# ============================================================================
# Stage 1 (Producer):  disk → cv2.imread → CPU tensor → raw_queue
# Stage 2 (Pinner):    raw_queue → .pin_memory() → ready_queue
# Consumer:            ready_queue → .to(device, non_blocking=True) → GPU
#
# The queues are bounded; producers block when the consumer is slow so RAM
# stays bounded.  Set PREFETCH_BATCHES=0 to fall back to synchronous loading.

# Number of batches to keep ready in the raw (disk-loaded) queue.
# Higher = more RAM used (~5 MB per batch at BS=4), but smoother GPU feeding.
PREFETCH_BATCHES = 10      # raw_queue capacity  (Stage 1 buffer)

# Parallel disk-loading threads (Stage 1 producers).
# 1 is usually sufficient on SSD/NVMe; increase to 2 on spinning HDDs.
PREFETCH_WORKERS = 1       # producer threads

# Pinning threads (Stage 2).  Each thread calls .pin_memory() on a batch
# so the GPU DMA transfer can proceed without involving the CPU.
# Set to 0 to skip pinning (CPU-only or debugging).
PREFETCH_PIN_WORKERS = 1   # pinner threads


# ============================================================================
# PATHS (Match dataset_generator_v2 output structure)
# ============================================================================

# Dataset root directory - base directory for all datasets
DATASET_ROOT = "/mnt/data/training/datasetNeu4kNeu"

# Dataset name (category)
# Options: 'master', 'universal', 'space', 'toon' (lowercase)
DEFAULT_DATASET_NAME = "master"

# Derived convenience path
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

# Enable Automatic Mixed Precision for faster training on Tesla P100.
# The P100 has native FP16 hardware (18.7 TFLOPS FP16 vs 9.3 TFLOPS FP32),
# so AMP delivers a real ~1.5–2× speedup and reduces activation memory by ~30-40%.
USE_AMP = True

# Enable gradient checkpointing (activation recomputation).
# Reduces activation memory by ~40% at the cost of ~10-15% compute overhead.
# Set to False to disable (higher VRAM usage, slightly faster training).
USE_CHECKPOINTING = True


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
        
        # Data loading pipeline
        'PREFETCH_BATCHES':      PREFETCH_BATCHES,
        'PREFETCH_WORKERS':      PREFETCH_WORKERS,
        'PREFETCH_PIN_WORKERS':  PREFETCH_PIN_WORKERS,
        
        # Paths
        'DATA_ROOT': DATA_ROOT,
        'DATASET_ROOT': DATASET_ROOT,
        'DEFAULT_DATASET_NAME': DEFAULT_DATASET_NAME,
        
        # Adaptive system
        'ADAPTIVE_LOSS_WEIGHTS': ADAPTIVE_LOSS_WEIGHTS,
        'ADAPTIVE_GRAD_CLIP': ADAPTIVE_GRAD_CLIP,
        'INITIAL_GRAD_CLIP': INITIAL_GRAD_CLIP,
    }
    
    # Add AMP setting (always include so training code can rely on its presence)
    config['USE_AMP'] = USE_AMP
    config['USE_CHECKPOINTING'] = USE_CHECKPOINTING
    
    return config


def print_config():
    """Print current configuration in a readable format."""
    print("\n" + "="*80)
    print("7-FRAME VSR CONFIGURATION (Tesla P100 Optimized)")
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
    
    print("\nDATA LOADING PIPELINE:")
    print(f"  Prefetch Batches (raw queue):  {PREFETCH_BATCHES}")
    print(f"  Producer Threads (disk I/O):   {PREFETCH_WORKERS}")
    print(f"  Pinner  Threads  (pin_memory): {PREFETCH_PIN_WORKERS}")
    
    print("\nDATASET PATHS:")
    print(f"  Dataset Root:           {DATASET_ROOT}")
    print(f"  Category (dataset_name): {DEFAULT_DATASET_NAME}")
    print(f"  Expected structure:")
    for size_key in ADAPTIVE_BATCH_CONFIG:
        print(f"    Training {size_key}:")
        print(f"      {DATASET_ROOT}/{DEFAULT_DATASET_NAME}/patches/{size_key}/GT/")
        print(f"      {DATASET_ROOT}/{DEFAULT_DATASET_NAME}/patches/{size_key}/LR_7frames/")
    print(f"    Validation:")
    for size_key in ADAPTIVE_BATCH_CONFIG:
        print(f"      {DATASET_ROOT}/{DEFAULT_DATASET_NAME}/val/{size_key}/GT/")

    print("\nADAPTIVE SYSTEM:")
    print(f"  Adaptive Loss Weights:  {ADAPTIVE_LOSS_WEIGHTS}")
    print(f"  Adaptive Grad Clip:     {ADAPTIVE_GRAD_CLIP}")
    print(f"  Initial Grad Clip:      {INITIAL_GRAD_CLIP}")

    print("\nPERFORMANCE:")
    print(f"  Mixed Precision (AMP):  {USE_AMP}")
    print(f"  Gradient Checkpointing: {USE_CHECKPOINTING}")

    print("\n" + "="*80)
    print("CONFIGURATION NOTES:")
    print("  - 7-frame model with 72 features and 26 blocks")
    print("  - Dataset structure matches dataset_generator_v2 output")
    print("  - Lowercase category names (master, universal, space, toon)")
    print("  - Size-specific directories: patches/{size_key}/ and val/{size_key}/")
    print("  - Validation LR files auto-found in patches/{size_key}/LR_7frames/")
    print("  - VGG perceptual loss enabled for quality")
    print("  - Gradient accumulation for per-size optimized effective batch sizes")
    print("="*80 + "\n")


if __name__ == '__main__':
    print_config()
