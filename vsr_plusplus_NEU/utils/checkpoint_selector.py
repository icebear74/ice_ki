"""
Interactive Checkpoint Selector

Shared module for checkpoint selection used by both training and standalone scripts.
Provides interactive UI for selecting from available checkpoints.
"""

import os
import sys

# ANSI colors
C_GREEN = "\033[92m"
C_CYAN = "\033[96m"
C_YELLOW = "\033[93m"
C_RESET = "\033[0m"


def select_checkpoint_interactive(checkpoint_mgr, auto_select_latest=False):
    """
    Interactive checkpoint selection with formatted display
    
    Args:
        checkpoint_mgr: CheckpointManager instance
        auto_select_latest: If True, automatically select latest without prompting
    
    Returns:
        dict: Selected checkpoint info with keys: path, step, type, quality, loss, date_str
        None: If no checkpoints available
    """
    all_checkpoints = checkpoint_mgr.list_checkpoints()
    
    if not all_checkpoints:
        print(f"{C_YELLOW}⚠️  No checkpoints found{C_RESET}")
        return None
    
    # Auto-select latest if requested
    if auto_select_latest:
        selected_ckpt = all_checkpoints[-1]
        print(f"{C_GREEN}✅ Auto-selected latest checkpoint: Step {selected_ckpt['step']:,}{C_RESET}")
        return selected_ckpt
    
    # Show detailed checkpoint selection menu
    print("=" * 100)
    print("AVAILABLE CHECKPOINTS (Last 10):")
    print("=" * 100)
    print(f"{'#':<4} {'Step':<12} {'Type':<12} {'Quality':<12} {'Loss':<10} {'Date':<18}")
    print("-" * 100)
    
    # Show last 10 checkpoints
    recent_checkpoints = all_checkpoints[-10:]
    for idx, ckpt in enumerate(recent_checkpoints, 1):
        step_display = f"{ckpt['step']:,}"
        type_display = ckpt['type']
        quality_display = f"{ckpt['quality']*100:.1f}%"
        loss_display = f"{ckpt['loss']:.4f}"
        date_display = ckpt['date_str']
        
        print(f"{idx:<4} {step_display:<12} {type_display:<12} {quality_display:<12} {loss_display:<10} {date_display:<18}")
    
    print("=" * 100)
    
    # User selection
    selection = input(f"\n{C_CYAN}Welchen Checkpoint laden? (Nummer 1-{len(recent_checkpoints)} oder Enter für neuesten): {C_RESET}").strip()
    
    if selection == "":
        # Use latest (last in list)
        selected_ckpt = all_checkpoints[-1]
        print(f"{C_GREEN}✅ Using latest checkpoint: Step {selected_ckpt['step']:,}{C_RESET}")
    else:
        try:
            choice_idx = int(selection)
            if 1 <= choice_idx <= len(recent_checkpoints):
                selected_ckpt = recent_checkpoints[choice_idx - 1]
                print(f"{C_GREEN}✅ Selected checkpoint: Step {selected_ckpt['step']:,} ({selected_ckpt['type']}){C_RESET}")
            else:
                print(f"{C_YELLOW}Invalid selection, using latest checkpoint{C_RESET}")
                selected_ckpt = all_checkpoints[-1]
        except ValueError:
            print(f"{C_YELLOW}Invalid input, using latest checkpoint{C_RESET}")
            selected_ckpt = all_checkpoints[-1]
    
    print()
    return selected_ckpt


def get_checkpoint_dir_from_config():
    """
    Get checkpoint directory from config following training path structure
    
    Returns:
        str: Path to checkpoint directory
    """
    try:
        # Try to import config from vsr_plusplus_NEU
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import config as cfg
        
        config = cfg.get_config()
        DATASET_ROOT = config.get('DATASET_ROOT', "/mnt/data/training/Dataset/Universal/Mastermodell")
        
        # Try to load runtime_config for dataset-specific paths
        runtime_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runtime_config.json")
        dataset_name = 'master'  # Default
        
        if os.path.exists(runtime_config_path):
            try:
                import json
                with open(runtime_config_path, 'r') as f:
                    rt_config = json.load(f)
                data_config = rt_config.get('data', {})
                DATASET_ROOT = data_config.get('root', DATASET_ROOT)
                dataset_name = data_config.get('dataset_name', 'master')
            except (json.JSONDecodeError, IOError, KeyError):
                pass
        
        # Dataset-specific checkpoint directory
        checkpoint_dir = os.path.join(DATASET_ROOT, dataset_name)
        
        return checkpoint_dir
    
    except (ImportError, AttributeError, KeyError) as e:
        # Fallback to default
        print(f"{C_YELLOW}⚠️  Could not load config, using default path: {e}{C_RESET}")
        return "/mnt/data/training/Dataset/Universal/Mastermodell/master"


def get_data_root_from_config():
    """
    Get DATA_ROOT from config
    
    Returns:
        str: Path to DATA_ROOT (for training data, validation, etc.)
    """
    try:
        # Try to import config
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import config as cfg
        
        config = cfg.get_config()
        return config.get('DATA_ROOT', "/mnt/data/training/Universal/Mastermodell/Learn")
    
    except (ImportError, AttributeError, KeyError) as e:
        # Fallback to default
        print(f"{C_YELLOW}⚠️  Could not load config, using default DATA_ROOT: {e}{C_RESET}")
        return "/mnt/data/training/Universal/Mastermodell/Learn"
