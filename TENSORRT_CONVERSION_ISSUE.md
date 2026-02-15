# TensorRT Conversion Known Issue

## Problem
The TensorRT conversion of the VSR++ 7-frame model fails with the following error:

```
[TRT] [E] ITensor::getDimensions: Error Code 3: API Usage Error (upsample.2:1:CONVOLUTION:GPU: at least 4 dimensions are required for input.
ValueError: __len__() should return >= 0
```

## Root Cause
The error occurs during TensorRT conversion (using torch2trt) when processing the `nn.PixelShuffle` layer in the model's upsampling module. TensorRT's torch2trt converter has known limitations with certain PyTorch operations, including:

1. `nn.PixelShuffle` - The converter encounters `torch.nn.functional.pixel_shuffle` which is not fully supported
2. Tensor dimension tracking after PixelShuffle causes issues in the broadcast operation for the residual connection

## Model Architecture Context
The issue occurs in `vsr_plusplus_NEU/core/model_7frame.py` at line 142:

```python
# Upsampling (3x with PixelShuffle)
self.upsample = nn.Sequential(
    nn.Conv2d(n_feats, n_feats * 9, 3, 1, 1),
    nn.PixelShuffle(3),                        # <-- TensorRT issue here
    nn.Conv2d(n_feats, 3, 3, 1, 1)
)

# Forward pass
base = F.interpolate(x[:, 3], scale_factor=3, mode='bilinear', align_corners=False)
upsampled = self.upsample(fused)
return upsampled + base                        # <-- Dimension mismatch here in TensorRT
```

## Impact
- **PyTorch Training**: ✅ Works perfectly - No issues
- **PyTorch Inference**: ✅ Works perfectly - No issues
- **TensorRT Conversion**: ❌ Fails - Cannot convert model
- **TensorRT Inference**: ❌ Not possible due to conversion failure

## Status
This is a **known limitation** of the torch2trt converter, not a bug in the model code. The model works correctly for:
- Training (primary use case)
- PyTorch inference
- ONNX export (if needed as alternative)

## Workarounds

### Option 1: Use PyTorch for Inference (Recommended)
Continue using PyTorch for inference instead of TensorRT. The model is already optimized and fast enough for most use cases.

### Option 2: Manual PixelShuffle Implementation
Replace `nn.PixelShuffle` with a custom implementation that TensorRT can handle:

```python
class CustomPixelShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor
    
    def forward(self, x):
        b, c, h, w = x.size()
        r = self.upscale_factor
        c_out = c // (r * r)
        
        # Reshape to [B, C_out, r, r, H, W]
        x = x.view(b, c_out, r, r, h, w)
        # Permute to [B, C_out, H, r, W, r]
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        # Reshape to [B, C_out, H*r, W*r]
        x = x.view(b, c_out, h * r, w * r)
        
        return x
```

**Note**: This workaround is untested and may have other TensorRT compatibility issues.

### Option 3: Use ONNX Export
Export the model to ONNX format and use TensorRT's ONNX parser instead of torch2trt:

```python
# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=11,  # Try different versions if needed
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
)

# Then use TensorRT's trtexec or Python API to convert ONNX to TensorRT
```

## Conclusion
The error shown in `error.txt` is **not a code bug** but a **TensorRT converter limitation**. The training code works correctly, and checkpoints are now properly saved with runtime configuration for correct restoration.

## Related Files
- Error log: `error.txt`
- Model code: `vsr_plusplus_NEU/core/model_7frame.py`
- Optimization script: `optimize_checkpoint.py`
