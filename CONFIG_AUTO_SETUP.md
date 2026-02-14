# Automatic Config Setup Fix

## Problem

When running `train.py`, users encountered an ImportError:
```
File "/mnt/data/ice_ki/vsr_plusplus_NEU/train.py", line 712, in <module>
    main()
File "/mnt/data/ice_ki/vsr_plusplus_NEU/train.py", line 43, in <module>
    import config as cfg
ImportError: No module named 'config'
```

### Root Cause

- `train.py` requires `config.py` to be imported
- But `config.py` is in `.gitignore` (user-specific configuration)
- Only `config.py.example` (template) and `config.py.active` (user's actual config) exist in the repository

## Solution

Added automatic config file creation with smart fallback logic in `train.py`.

### How It Works

**Before importing config, the script now:**

1. **Checks if `config.py` exists**
   - If yes: proceeds with import ✓

2. **If not, tries `config.py.active`** (user's pushed config)
   ```
   ⚠ config.py not found, using config.py.active
   ✓ Created config.py from config.py.active
   ```

3. **If not, tries `config.py.example`** (template)
   ```
   ⚠ config.py not found, using config.py.example
   ✓ Created config.py from config.py.example
     Please edit config.py to match your setup!
   ```

4. **If neither exists: clear error message**
   ```
   ❌ ERROR: No configuration file found!
      Expected one of:
      - /path/to/vsr_plusplus_NEU/config.py
      - /path/to/vsr_plusplus_NEU/config.py.active
      - /path/to/vsr_plusplus_NEU/config.py.example
   ```

## Code Changes

**File:** `vsr_plusplus_NEU/train.py`  
**Lines:** 41-67

```python
# Smart fallback: try config.py, then config.py.active, then config.py.example
import shutil

_config_dir = os.path.dirname(os.path.abspath(__file__))
_config_path = os.path.join(_config_dir, 'config.py')
_config_active_path = os.path.join(_config_dir, 'config.py.active')
_config_example_path = os.path.join(_config_dir, 'config.py.example')

if not os.path.exists(_config_path):
    # config.py doesn't exist, try to create it from .active or .example
    if os.path.exists(_config_active_path):
        print(f"{C_YELLOW}⚠ config.py not found, using config.py.active{C_RESET}")
        shutil.copy(_config_active_path, _config_path)
        print(f"{C_GREEN}✓ Created config.py from config.py.active{C_RESET}")
    elif os.path.exists(_config_example_path):
        print(f"{C_YELLOW}⚠ config.py not found, using config.py.example{C_RESET}")
        shutil.copy(_config_example_path, _config_path)
        print(f"{C_GREEN}✓ Created config.py from config.py.example{C_RESET}")
        print(f"{C_YELLOW}  Please edit config.py to match your setup!{C_RESET}")
    else:
        print(f"{C_RED}❌ ERROR: No configuration file found!{C_RESET}")
        print(f"{C_RED}   Expected one of:{C_RESET}")
        print(f"{C_RED}   - {_config_path}{C_RESET}")
        print(f"{C_RED}   - {_config_active_path}{C_RESET}")
        print(f"{C_RED}   - {_config_example_path}{C_RESET}")
        sys.exit(1)

import config as cfg
```

## Benefits

### 1. **Zero Manual Setup**
- No need to manually copy `config.py.example` to `config.py`
- No need to manually copy `config.py.active` to `config.py`
- Script does it automatically

### 2. **Uses User's Active Config**
- If user pushed their `config.py.active`, it's automatically used
- No risk of using outdated example config

### 3. **Clear Error Messages**
- If something goes wrong, user knows exactly what files are missing
- Shows full paths for easy troubleshooting

### 4. **Backward Compatible**
- If `config.py` already exists, nothing changes
- Works with existing setups

### 5. **Development Friendly**
- Developers can use `config.py.active` to share working configs
- Template (`config.py.example`) still available for reference

## Usage Scenarios

### Scenario 1: Fresh Clone (with config.py.active)
```bash
git clone <repo>
cd vsr_plusplus_NEU
python train.py
```
Output:
```
⚠ config.py not found, using config.py.active
✓ Created config.py from config.py.active
VSR++ Training System - Manual Configuration
...
```

### Scenario 2: Fresh Clone (without config.py.active)
```bash
git clone <repo>
cd vsr_plusplus_NEU
python train.py
```
Output:
```
⚠ config.py not found, using config.py.example
✓ Created config.py from config.py.example
  Please edit config.py to match your setup!
VSR++ Training System - Manual Configuration
...
```

### Scenario 3: config.py already exists
```bash
python train.py
```
Output:
```
VSR++ Training System - Manual Configuration
...
```
(No warnings, uses existing config.py)

### Scenario 4: No config files at all (error)
```bash
python train.py
```
Output:
```
❌ ERROR: No configuration file found!
   Expected one of:
   - /path/to/vsr_plusplus_NEU/config.py
   - /path/to/vsr_plusplus_NEU/config.py.active
   - /path/to/vsr_plusplus_NEU/config.py.example
```
(Script exits with error code 1)

## Testing

### Tested Scenarios
- ✅ config.py doesn't exist, config.py.active exists → auto-created ✓
- ✅ config.py doesn't exist, only config.py.example exists → auto-created ✓
- ✅ config.py already exists → no changes ✓
- ✅ Import works after auto-creation ✓
- ✅ No syntax errors ✓

### Manual Testing
```bash
# Test 1: Remove config.py and test auto-creation from .active
cd vsr_plusplus_NEU
rm -f config.py
python train.py
# Should see: "⚠ config.py not found, using config.py.active"

# Test 2: Remove config.py and config.py.active, test .example fallback
rm -f config.py config.py.active
python train.py
# Should see: "⚠ config.py not found, using config.py.example"

# Restore config.py.active
git checkout config.py.active
```

## Impact

**Before Fix:**
- ❌ Training failed with ImportError
- ❌ Users had to manually create config.py
- ❌ Confusing error message

**After Fix:**
- ✅ Training starts automatically
- ✅ Uses user's active configuration
- ✅ Clear, helpful messages
- ✅ Zero manual setup required

## Related Files

- `vsr_plusplus_NEU/train.py` - Main training script (modified)
- `vsr_plusplus_NEU/config.py.example` - Configuration template
- `vsr_plusplus_NEU/config.py.active` - User's active configuration
- `vsr_plusplus_NEU/.gitignore` - Contains `config.py` (user-specific)

## Future Improvements

Possible enhancements:
1. Add `--config` command-line argument to specify custom config file
2. Add validation of config values before importing
3. Add interactive config wizard for first-time setup
4. Add config file versioning/migration support

## Notes

- The auto-created `config.py` is still ignored by git (in `.gitignore`)
- Users can edit `config.py` after it's created
- `config.py.active` can be updated and pushed to share configs
- `config.py.example` should always reflect the latest template

---

**Status:** ✅ FIXED  
**Commit:** 6674e18  
**Date:** 2026-02-14
