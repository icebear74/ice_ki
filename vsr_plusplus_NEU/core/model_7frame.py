"""
7-Frame Bidirectional VSR Model
MATCHES original VSRBidirectional_3x architecture exactly for realistic memory testing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class AttentionGate(nn.Module):
    """
    Spatial Attention Gate for filtering skip connection features.
    Learns WHICH pixels of the skip features are relevant (e.g. suppresses motion blur/grain).

    gate_feat: the "query" - what the model currently knows (propagated features)
    skip_feat: the "value" - the raw frame features from the skip connection

    Output: filtered skip_feat, scaled by learned spatial attention map (0..1 per pixel)
    """
    def __init__(self, n_feats):
        super().__init__()
        # Combine gate signal + skip signal -> attention map
        self.gate_conv = nn.Conv2d(n_feats * 2, n_feats, 1)
        self.gate_relu = nn.ReLU(inplace=False)
        self.gate_out  = nn.Conv2d(n_feats, 1, 1)   # Single-channel spatial mask
        self.sigmoid   = nn.Sigmoid()

        # Tracking for WebUI
        self.last_gate_mean = 0.0
        self.last_gate_min  = 0.0
        self.last_gate_max  = 1.0

    def forward(self, gate_feat, skip_feat):
        combined = torch.cat([gate_feat, skip_feat], dim=1)
        gate_map = self.gate_conv(combined)
        gate_map = self.gate_relu(gate_map)
        gate_map = self.gate_out(gate_map)
        gate_map = self.sigmoid(gate_map)  # Shape: [B, 1, H, W]

        # Track gate statistics
        with torch.no_grad():
            self.last_gate_mean = gate_map.mean().item()
            self.last_gate_min  = gate_map.min().item()
            self.last_gate_max  = gate_map.max().item()

        return skip_feat * gate_map  # Filtered skip features


class ResidualBlock(nn.Module):
    """
    Residual block with optional Attention Gate on the skip connection.

    When use_attention=True:
        - The skip (residual) is NOT blindly added
        - Instead, an AttentionGate filters the skip based on what the block learned
        - This prevents film grain and blur from being "copied" through skip connections

    Backward compatible: use_attention=False behaves exactly like the original.
    """
    def __init__(self, n_feats, use_checkpointing=False, use_attention=True):
        super().__init__()
        self.conv1 = nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        self.relu  = nn.LeakyReLU(0.1, inplace=False)  # LeakyReLU like original
        self.conv2 = nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        self.use_checkpointing = use_checkpointing
        self.use_attention     = use_attention
        self.last_activity     = 0.0

        if use_attention:
            self.attn_gate = AttentionGate(n_feats)

    def _forward_impl(self, x):
        """Internal forward computation, separated for gradient checkpointing support."""
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        if self.use_attention:
            # Gate controls HOW MUCH of the residual to add back
            # gate_feat = out (what the block learned), skip_feat = residual (raw input)
            gated_residual = self.attn_gate(out, residual)
            out = out + gated_residual
        else:
            out = residual + out

        return out

    def forward(self, x):
        if self.use_checkpointing and self.training:
            out = checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            out = self._forward_impl(x)

        # Track activity
        self.last_activity = out.detach().abs().mean().item()

        return out


# ---------------------------------------------------------------------------
# Legacy FusionBlock — kept for loading old checkpoints (do not use in new
# training runs; use GatedFusionBlock instead).
# ---------------------------------------------------------------------------
class FusionBlock(nn.Module):
    """
    FusionBlock v2: Mit Residual Skip-Connection (1x1).

    Der Skip-Pfad (1x1 Conv, kein Bias) projiziert in_feats → out_feats direkt.
    Der Haupt-Pfad (3x3 → ReLU → 1x1) lernt nur noch die Korrektur.
    Bewährtes ResNet-Muster: stabilere, schnellere Konvergenz.

    Trackt Aktivität aller drei Conv-Layer getrennt für WebUI-Visualisierung.

    .. deprecated::
        Replaced by :class:`GatedFusionBlock` for new training runs.
    """
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.conv3x3 = nn.Conv2d(in_feats, out_feats, 3, 1, 1)
        self.relu    = nn.LeakyReLU(0.1, inplace=False)
        self.conv1x1 = nn.Conv2d(out_feats, out_feats, 1)
        # Skip: projiziert in_feats → out_feats (kein Bias, kein räumlicher Kontext)
        # Entspricht dem Dimensions-Anpassungs-Skip aus ResNet (Option B)
        self.skip    = nn.Conv2d(in_feats, out_feats, 1, bias=False)
        self.last_activity_3x3  = 0.0
        self.last_activity_1x1  = 0.0
        self.last_activity_skip = 0.0

    def forward(self, x):
        # Skip-Pfad: direkte Projektion (günstig, stabil)
        identity = self.skip(x)
        self.last_activity_skip = identity.detach().abs().mean().item()

        # Haupt-Pfad: lernt nur noch die Korrektur auf identity
        out = self.conv3x3(x)
        self.last_activity_3x3 = out.detach().abs().mean().item()
        out = self.relu(out)
        out = self.conv1x1(out)
        self.last_activity_1x1 = out.detach().abs().mean().item()

        return out + identity


class GatedFusionBlock(nn.Module):
    """
    Gated Fusion Block - replaces the original FusionBlock.

    Unlike the original (conv3x3 -> relu -> conv1x1), this block adds a GATE branch:
    - Main branch: conv3x3 -> relu -> conv1x1  (learns WHAT to output)
    - Gate branch: conv1x1 -> sigmoid          (learns HOW MUCH to output, pixel-wise)

    Output = main_branch * gate_branch

    This suppresses ghosting artifacts and motion blur by learning to mask them out.
    Tracks activity of main, gate, and final output separately for WebUI.
    """
    def __init__(self, in_feats, out_feats):
        super().__init__()
        # Main branch (same as original FusionBlock)
        self.conv3x3 = nn.Conv2d(in_feats, out_feats, 3, 1, 1)
        self.relu    = nn.LeakyReLU(0.1, inplace=False)
        self.conv1x1 = nn.Conv2d(out_feats, out_feats, 1)

        # Gate branch (NEW)
        self.gate    = nn.Sequential(
            nn.Conv2d(in_feats, out_feats, 1),
            nn.Sigmoid()
        )

        # Activity tracking (backward compatible with existing WebUI)
        self.last_activity_3x3  = 0.0
        self.last_activity_1x1  = 0.0
        self.last_activity_gate = 0.0

    def forward(self, x):
        # Main branch
        out = self.conv3x3(x)
        self.last_activity_3x3 = out.detach().abs().mean().item()

        out = self.relu(out)
        out = self.conv1x1(out)
        self.last_activity_1x1 = out.detach().abs().mean().item()

        # Gate branch
        gate_map = self.gate(x)
        self.last_activity_gate = gate_map.detach().mean().item()

        return out * gate_map


class VSRBidirectional_7frames_3x(nn.Module):
    """
    7-Frame Bidirectional VSR Model - EXACT MATCH to original training architecture
    
    Input: [B, 7, 3, H, W] (7 frames)
    Output: [B, 3, H*3, W*3] (upscaled center frame)
    
    Architecture matches VSRBidirectional_3x for realistic memory measurements.
    """
    def __init__(self, n_feats=72, n_blocks=26, use_checkpointing=False):
        super().__init__()
        self.n_feats = n_feats
        self.n_blocks = n_blocks

        half_blocks = max(1, n_blocks // 2)

        # 1. Feature Extraction
        self.feat_extract = nn.Conv2d(3, n_feats, 3, 1, 1)

        # 2. Fusion layers - NOW GATED (suppresses ghosting/motion blur)
        self.backward_fuse = GatedFusionBlock(n_feats * 2, n_feats)
        self.forward_fuse  = GatedFusionBlock(n_feats * 2, n_feats)

        # 3. Propagation Trunks - NOW WITH ATTENTION GATES on skip connections
        self.backward_trunk = nn.ModuleList([
            ResidualBlock(n_feats, use_checkpointing=use_checkpointing, use_attention=True)
            for _ in range(half_blocks)
        ])
        self.forward_trunk = nn.ModuleList([
            ResidualBlock(n_feats, use_checkpointing=use_checkpointing, use_attention=True)
            for _ in range(half_blocks)
        ])

        # 4. Final Fusion - NOW GATED
        self.fusion = GatedFusionBlock(n_feats * 2, n_feats)

        # 5. Upsampling (3x with PixelShuffle) - unchanged
        self.upsample = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * 9, 3, 1, 1),
            nn.PixelShuffle(3),
            nn.Conv2d(n_feats, 3, 3, 1, 1)
        )
        
    def forward(self, x):
        """
        Forward pass matching original architecture
        
        Args:
            x: Input tensor [B, 7, 3, H, W]
            
        Returns:
            Output tensor [B, 3, H*3, W*3]
        """
        B, T, C, H, W = x.size()
        
        # Extract features from all 7 frames
        feats = self.feat_extract(x.view(-1, C, H, W))
        feats = feats.view(B, T, self.n_feats, H, W)
        
        # Initialize with Frame 4 (center frame, index 3)
        center_feat = feats[:, 3].clone()
        
        # Backward propagation: F4 → F5 → F6
        back_prop = center_feat
        for i in [4, 5, 6]:
            # Fuse THEN process through trunk (like original)
            fused = self.backward_fuse(torch.cat([back_prop, feats[:, i]], dim=1))
            for block in self.backward_trunk:
                fused = block(fused)
            back_prop = fused
        
        # Forward propagation: F4 → F3 → F2 → F1
        forw_prop = center_feat
        for i in [2, 1, 0]:
            # Fuse THEN process through trunk (like original)
            fused = self.forward_fuse(torch.cat([forw_prop, feats[:, i]], dim=1))
            for block in self.forward_trunk:
                fused = block(fused)
            forw_prop = fused
        
        # Fuse bidirectional features
        fused = self.fusion(torch.cat([back_prop, forw_prop], dim=1))
        
        # Upsampling with residual connection (CRITICAL - like original!)
        base = F.interpolate(x[:, 3], scale_factor=3, mode='bilinear', align_corners=False)
        upsampled = self.upsample(fused)
        
        return upsampled + base
    
    def get_layer_activity(self):
        """
        Returns activity levels for all blocks including fusion layers and gate statistics.

        Returns:
            Dict with activities:
            {
                'backward_trunk': [list of ResidualBlock activities],
                'backward_trunk_gate': [list of {mean, min, max} dicts per block with use_attention],
                'backward_fuse': [3x3 activity, 1x1 activity, gate activity],
                'forward_trunk': [list of ResidualBlock activities],
                'forward_trunk_gate': [list of {mean, min, max} dicts per block with use_attention],
                'forward_fuse': [3x3 activity, 1x1 activity, gate activity],
                'fusion': [3x3 activity, 1x1 activity, gate activity]
            }
        """
        backward_activities = []
        backward_gate_stats = []
        for block in self.backward_trunk:
            backward_activities.append(block.last_activity)
            if block.use_attention:
                backward_gate_stats.append({
                    'mean': block.attn_gate.last_gate_mean,
                    'min':  block.attn_gate.last_gate_min,
                    'max':  block.attn_gate.last_gate_max,
                })

        forward_activities = []
        forward_gate_stats = []
        for block in self.forward_trunk:
            forward_activities.append(block.last_activity)
            if block.use_attention:
                forward_gate_stats.append({
                    'mean': block.attn_gate.last_gate_mean,
                    'min':  block.attn_gate.last_gate_min,
                    'max':  block.attn_gate.last_gate_max,
                })

        return {
            'backward_trunk':      backward_activities,
            'backward_trunk_gate': backward_gate_stats,
            'backward_fuse':       [self.backward_fuse.last_activity_3x3, self.backward_fuse.last_activity_1x1, getattr(self.backward_fuse, 'last_activity_gate', getattr(self.backward_fuse, 'last_activity_skip', 0.0))],
            'forward_trunk':       forward_activities,
            'forward_trunk_gate':  forward_gate_stats,
            'forward_fuse':        [self.forward_fuse.last_activity_3x3, self.forward_fuse.last_activity_1x1, getattr(self.forward_fuse, 'last_activity_gate', getattr(self.forward_fuse, 'last_activity_skip', 0.0))],
            'fusion':              [self.fusion.last_activity_3x3, self.fusion.last_activity_1x1, getattr(self.fusion, 'last_activity_gate', getattr(self.fusion, 'last_activity_skip', 0.0))],
        }

