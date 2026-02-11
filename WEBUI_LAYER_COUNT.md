# WebUI Layer Visualization - 7-Frame Model

## Expected Layer Count

When using the **7-frame model** (`vsr_plusplus_NEU/core/model_7frame.py`), you should see **32 layers** total in the WebUI.

### Layer Breakdown

| Layer Type | Count | Description |
|------------|-------|-------------|
| Backward Trunk | 13 | ResidualBlock layers (backward propagation) |
| Backward Fusion 3x3 | 1 | FusionBlock 3x3 conv (spatial context) |
| Backward Fusion 1x1 | 1 | FusionBlock 1x1 conv (channel gating) |
| Forward Trunk | 13 | ResidualBlock layers (forward propagation) |
| Forward Fusion 3x3 | 1 | FusionBlock 3x3 conv (spatial context) |
| Forward Fusion 1x1 | 1 | FusionBlock 1x1 conv (channel gating) |
| Final Fusion 3x3 | 1 | FusionBlock 3x3 conv (spatial context) |
| Final Fusion 1x1 | 1 | FusionBlock 1x1 conv (channel gating) |
| **TOTAL** | **32** | All layers with activity tracking |

### Activity Structure

The `get_layer_activity()` method returns:

```python
{
    'backward_trunk': [13 float values],      # Activity of 13 ResidualBlocks
    'backward_fuse': [3x3_act, 1x1_act],      # 2 values: 3x3 and 1x1 activities
    'forward_trunk': [13 float values],       # Activity of 13 ResidualBlocks
    'forward_fuse': [3x3_act, 1x1_act],       # 2 values: 3x3 and 1x1 activities
    'fusion': [3x3_act, 1x1_act]              # 2 values: 3x3 and 1x1 activities
}
```

### What's New? ⭐

The **FusionBlock layers** are tracked separately:
- **3x3 convolution**: Provides spatial context (sees neighboring pixels)
- **1x1 convolution**: Provides channel gating (feature selection)

This separate tracking allows you to:
- See which layer is more active (3x3 spatial vs 1x1 gating)
- Debug issues with specific parts of the fusion block
- Understand the model's internal behavior better

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
5. Each FusionBlock should show 2 values (as a list), not 1

## Notes

- The 5-frame models remain unchanged and will show their original layer counts
- Only `vsr_plusplus_NEU/core/model_7frame.py` has the enhanced FusionBlocks
- Activity values range from 0.0 to higher values depending on the data being processed
- Separate tracking does NOT change model quality, only visualization detail
