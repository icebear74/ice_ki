# WebUI Layer Visualization - 7-Frame Model

## Expected Layer Count

When using the **7-frame model** (`vsr_plusplus_NEU/core/model_7frame.py`), you should see **29 layers** total in the WebUI.

### Layer Breakdown

| Layer Type | Count | Description |
|------------|-------|-------------|
| Backward Trunk | 13 | ResidualBlock layers (backward propagation) |
| Backward Fusion | 1 | **FusionBlock** (NEW - enhanced with 3x3+1x1 conv) |
| Forward Trunk | 13 | ResidualBlock layers (forward propagation) |
| Forward Fusion | 1 | **FusionBlock** (NEW - enhanced with 3x3+1x1 conv) |
| Final Fusion | 1 | **FusionBlock** (NEW - enhanced with 3x3+1x1 conv) |
| **TOTAL** | **29** | All layers with activity tracking |

### Activity Structure

The `get_layer_activity()` method returns:

```python
{
    'backward_trunk': [13 float values],  # Activity of 13 ResidualBlocks
    'backward_fuse': float,               # Activity of backward FusionBlock
    'forward_trunk': [13 float values],   # Activity of 13 ResidualBlocks
    'forward_fuse': float,                # Activity of forward FusionBlock
    'fusion': float                       # Activity of final FusionBlock
}
```

### What's New? ⭐

The **3 FusionBlock layers** (backward_fuse, forward_fuse, fusion) are NEW:
- Previously: Simple 1x1 convolutions (no spatial awareness)
- Now: FusionBlock with 3x3 convolution + LeakyReLU + 1x1 convolution
- Benefit: Better ghosting/shadow suppression and scene cut handling

### Configuration

Model is created with:
- `n_feats = 72` (number of feature channels)
- `n_blocks = 26` (total ResidualBlocks, split into 13 + 13)

Formula: `half_blocks = max(1, n_blocks // 2) = 13`

## How to Verify

If the WebUI shows a different number of layers:
1. Check that you're using the 7-frame model (`model_7frame.py`)
2. Verify the model configuration (n_blocks=26 gives 13+13 trunk blocks)
3. Ensure the model's `get_layer_activity()` method is being called
4. Check WebUI code for how it processes the activity dictionary

## Notes

- The 5-frame models remain unchanged and will show their original layer counts
- Only `vsr_plusplus_NEU/core/model_7frame.py` has the enhanced FusionBlocks
- Activity values range from 0.0 to higher values depending on the data being processed
