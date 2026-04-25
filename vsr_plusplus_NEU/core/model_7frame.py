"""
7-Frame Bidirectional VSR Model — P4-Optimized Architecture v2

Improvements over v1:
    1. AttentionGate on ResidualBlock skip connections:
       Filters the skip connection based on what the block processed.
       Prevents film grain and motion blur from being copied through residual paths.

    2. GatedFusionBlock (replaces FusionBlock):
       Adds a pixel-wise gate: output = main * gate.
       Suppresses ghosting, motion blur, and irrelevant frame content.

    3. TemporalAlignBlock (NEW):
       Before fusing neighbor frame features, they are aligned to the propagated
       features using a learned offset field. This compensates for motion between
       frames so that FusionBlock only needs to combine content, not correct motion.
"""

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class AttentionGate(nn.Module):
    """
    Spatial Attention Gate for filtering skip connection features.

    Learns WHICH pixels of the skip features are relevant.
    Suppresses motion blur, film grain, and ghosting artifacts.

    Args:
        gate_feat: query - what the model currently knows (propagated features)  [B, C, H, W]
        skip_feat: value - raw frame features from skip connection               [B, C, H, W]

    Returns:
        Filtered skip_feat, scaled by spatial attention map in range [0, 1]
    """
    def __init__(self, n_feats):
        super().__init__()
        self.gate_conv = nn.Conv2d(n_feats * 2, n_feats, 1)
        self.gate_relu = nn.ReLU(inplace=False)
        self.gate_out  = nn.Conv2d(n_feats, 1, 1)
        self.sigmoid   = nn.Sigmoid()
        # WebUI tracking
        self.last_gate_mean = 0.0
        self.last_gate_min  = 0.0
        self.last_gate_max  = 1.0

    def forward(self, gate_feat, skip_feat):
        combined = torch.cat([gate_feat, skip_feat], dim=1)
        gate_map = self.gate_conv(combined)
        gate_map = self.gate_relu(gate_map)
        gate_map = self.gate_out(gate_map)
        gate_map = self.sigmoid(gate_map)  # [B, 1, H, W]
        with torch.no_grad():
            self.last_gate_mean = gate_map.mean().item()
            self.last_gate_min  = gate_map.min().item()
            self.last_gate_max  = gate_map.max().item()
        return skip_feat * gate_map


class TemporalAlignBlock(nn.Module):
    """
    Learned Temporal Alignment Block.

    Aligns a neighbor frame's features (src_feat) to the reference frame's features
    (ref_feat) by learning a spatial offset field. This is a lightweight alternative
    to optical flow: instead of computing explicit flow, the block learns a
    correlation-based offset map and uses grid_sample to warp the source features.

    Architecture:
        1. Correlation layer: compare ref and src features to find similarity
        2. Offset conv: predict (dx, dy) offset per pixel from correlation
        3. Grid sample: warp src_feat using predicted offsets

    The warped src_feat is then spatially aligned to ref_feat, so the subsequent
    FusionBlock only needs to combine content — not correct for motion.

    Args:
        n_feats (int): Number of feature channels
        max_offset (float): Maximum displacement as fraction of feature map size (default: 0.2)

    WebUI tracking:
        last_flow_magnitude: mean absolute offset magnitude (0 = no motion, 1 = max motion)
    """
    def __init__(self, n_feats, max_offset=0.2):
        super().__init__()
        self.max_offset = max_offset

        # Step 1: Correlation — compare ref and src to find motion cues
        # Input: cat(ref, src) = 2*n_feats channels
        self.corr_conv = nn.Sequential(
            nn.Conv2d(n_feats * 2, n_feats, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(n_feats, n_feats // 2, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=False),
        )

        # Step 2: Predict offset field (2 channels = dx, dy)
        self.offset_conv = nn.Conv2d(n_feats // 2, 2, 3, 1, 1)

        # Bug 6 fix: use small Kaiming init instead of zero init.
        # Zero init causes near-zero gradients through grid_sample near the
        # identity mapping, so offsets never learn.  A small but non-zero
        # initialisation gives the gradient a path to grow from.
        nn.init.kaiming_normal_(self.offset_conv.weight, a=0.1, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.zeros_(self.offset_conv.bias)
        # Scale weights down so offsets start near-identity but remain learnable.
        with torch.no_grad():
            self.offset_conv.weight.mul_(0.01)

        # Bug 6 fix: explicit Kaiming init for correlation layers to ensure
        # good gradient flow into the offset prediction head.
        for m in self.corr_conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.1, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # WebUI tracking
        self.last_flow_magnitude = 0.0

    def forward(self, ref_feat, src_feat):
        """
        Args:
            ref_feat: reference (propagated) features [B, C, H, W]
            src_feat: source (neighbor frame) features [B, C, H, W]

        Returns:
            aligned_src: src_feat warped to align with ref_feat [B, C, H, W]
        """
        B, C, H, W = ref_feat.shape

        # Predict offset field
        corr = self.corr_conv(torch.cat([ref_feat, src_feat], dim=1))
        offset = self.offset_conv(corr)  # [B, 2, H, W]
        offset = torch.tanh(offset) * self.max_offset  # Clamp to [-max_offset, +max_offset]

        # Track motion magnitude for WebUI
        with torch.no_grad():
            self.last_flow_magnitude = offset.abs().mean().item()

        # Build sampling grid: base grid + learned offset
        # base grid has values in [-1, 1] (normalized device coordinates)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=ref_feat.device),
            torch.linspace(-1, 1, W, device=ref_feat.device),
            indexing='ij'
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1)   # [H, W, 2], float32
        base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2]

        # offset is [B, 2, H, W] → rearrange to [B, H, W, 2]
        offset_grid = offset.permute(0, 2, 3, 1)

        # Warp in float32 for FP16 compatibility:
        # • torch.linspace always returns float32, so base_grid is float32.
        # • F.grid_sample requires input and grid to have the same dtype.
        # • Do the warp in float32 regardless of the model's working dtype
        #   and cast the result back to the original dtype afterwards.
        sample_grid = (base_grid + offset_grid.float()).clamp(-1, 1)  # float32

        aligned_src = F.grid_sample(
            src_feat.float(),   # upcast to float32 if fp16
            sample_grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )

        return aligned_src.to(ref_feat.dtype)  # restore original dtype (fp16 or fp32)


class GatedFusionBlock(nn.Module):
    """
    Gated Fusion Block — upgraded replacement for FusionBlock.

    Adds a gating mechanism: output = main_branch * gate_branch
    - Main branch: conv3x3 → relu → conv1x1  (learns WHAT to output)
    - Gate branch: conv1x1 → sigmoid          (learns HOW MUCH, pixel-wise)

    Suppresses ghosting and motion blur artifacts.
    Backward compatible with FusionBlock tracking API (last_activity_3x3, last_activity_1x1).
    """
    def __init__(self, in_feats, out_feats):
        super().__init__()
        # Main branch
        self.conv3x3 = nn.Conv2d(in_feats, out_feats, 3, 1, 1)
        self.relu    = nn.LeakyReLU(0.1, inplace=False)
        self.conv1x1 = nn.Conv2d(out_feats, out_feats, 1)
        # Gate branch
        self.gate    = nn.Sequential(
            nn.Conv2d(in_feats, out_feats, 1),
            nn.Sigmoid()
        )
        # WebUI tracking (backward compatible)
        self.last_activity_3x3  = 0.0
        self.last_activity_1x1  = 0.0
        self.last_activity_gate = 0.0

    def forward(self, x):
        out = self.conv3x3(x)
        self.last_activity_3x3 = out.detach().abs().mean().item()
        out = self.relu(out)
        out = self.conv1x1(out)
        self.last_activity_1x1 = out.detach().abs().mean().item()
        gate_map = self.gate(x)
        self.last_activity_gate = gate_map.detach().mean().item()
        return out * gate_map


class ResidualBlock(nn.Module):
    """
    Residual block with optional Attention Gate on the skip connection.

    When use_attention=True (default for new training):
        - AttentionGate filters the skip (residual) based on processed features
        - Prevents film grain and blur from being blindly copied via skip connection

    When use_attention=False:
        - Behaves exactly like the original ResidualBlock (backward compatible)

    Supports gradient checkpointing for VRAM savings on Tesla P4.
    """
    def __init__(self, n_feats, use_checkpointing=False, use_attention=True):
        super().__init__()
        self.conv1 = nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        self.relu  = nn.LeakyReLU(0.1, inplace=False)
        self.conv2 = nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        self.use_checkpointing = use_checkpointing
        self.use_attention     = use_attention
        self.last_activity     = 0.0
        if use_attention:
            self.attn_gate = AttentionGate(n_feats)

    def _forward_impl(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        if self.use_attention:
            gated_residual = self.attn_gate(out, residual)
            out = out + gated_residual
        else:
            out = out + residual
        return out

    def forward(self, x):
        if self.use_checkpointing and self.training:
            out = checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            out = self._forward_impl(x)
        self.last_activity = out.detach().abs().mean().item()
        return out


class FusionBlock(nn.Module):
    """
    LEGACY: Original fusion block — kept for loading old checkpoints.
    For new training, use GatedFusionBlock instead.
    """
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.conv3x3 = nn.Conv2d(in_feats, out_feats, 3, 1, 1)
        self.relu    = nn.LeakyReLU(0.1, inplace=False)
        self.conv1x1 = nn.Conv2d(out_feats, out_feats, 1)
        self.last_activity_3x3 = 0.0
        self.last_activity_1x1 = 0.0

    def forward(self, x):
        out = self.conv3x3(x)
        self.last_activity_3x3 = out.detach().abs().mean().item()
        out = self.relu(out)
        out = self.conv1x1(out)
        self.last_activity_1x1 = out.detach().abs().mean().item()
        return out


class VSRBidirectional_7frames_3x(nn.Module):
    """
    7-Frame Bidirectional VSR Model — P4-Optimized Architecture v2

    Input:  [B, 7, 3, H, W]   (7 LR frames, F1..F7)
    Output: [B, 3, H*3, W*3]  (3x upscaled center frame F4)

    ── Which frames does each component touch? ───────────────────────────────

        F1   F2   F3  [F4]  F5   F6   F7
        │    │    │    │    │    │    │
        └────┴────┴───►│◄───┴────┴────┘
                  feat_extract         ← ALL 7 frames (single shared Conv2d)

        F4 is center (starting point, never passed through fuse/align directly)

        Backward direction — processes F5, F6, F7 (3 frames AFTER center):
            F5 → TemporalAlignBlock + GatedFusionBlock + trunk
            F6 → TemporalAlignBlock + GatedFusionBlock + trunk
            F7 → TemporalAlignBlock + GatedFusionBlock + trunk

        Forward direction — processes F3, F2, F1 (3 frames BEFORE center):
            F3 → TemporalAlignBlock + GatedFusionBlock + trunk
            F2 → TemporalAlignBlock + GatedFusionBlock + trunk
            F1 → TemporalAlignBlock + GatedFusionBlock + trunk

        Final fusion — combines bidirectional results:
            GatedFusionBlock(cat[back_prop, forw_prop])

    ── Component scope summary ───────────────────────────────────────────────

        feat_extract       : ALL 7 frames (F1–F7)
        TemporalAlignBlock : 6 neighbor frames (F5,F6,F7 backward + F3,F2,F1 forward)
        GatedFusionBlock   : 6 neighbor frames + 1 final fusion = 7 fusion steps total
        AttentionGate      : inside every ResidualBlock (trunk), fired 6×n_blocks times
        Center frame F4    : starting point only — features never go through fuse/align

    ── Improvements over v1 ──────────────────────────────────────────────────

    1. TemporalAlignBlock (NEW):
       Before fusing neighbor frame features, they are ALIGNED to the propagated
       features using a learned offset field. This compensates for motion between
       frames so that GatedFusionBlock only needs to combine content, not correct motion.

    2. GatedFusionBlock (NEW):
       Replaces FusionBlock. Adds a pixel-wise gate: output = main * gate.
       Suppresses ghosting, motion blur, and irrelevant frame content.

    3. AttentionGate on ResidualBlock skip connections (NEW):
       Filters the skip connection based on what the block processed.
       Prevents film grain and motion blur from being copied through residual paths.

    ── Data flow ─────────────────────────────────────────────────────────────

        [7 frames] → feat_extract → [7 feature maps]

        back_prop = forward_prop = center_feat = feats[:, 3]

        Backward: F4 → F5 → F6 → F7
            for each neighbor i in [4, 5, 6]:
                aligned_i = backward_align(back_prop, feats[:, i])   ← TemporalAlign
                fused = backward_fuse(cat([back_prop, aligned_i]))    ← GatedFusion
                back_prop = trunk(fused)                              ← AttentionGate inside

        Forward: F4 → F3 → F2 → F1
            for each neighbor i in [2, 1, 0]:
                aligned_i = forward_align(forw_prop, feats[:, i])    ← TemporalAlign
                fused = forward_fuse(cat([forw_prop, aligned_i]))     ← GatedFusion
                forw_prop = trunk(fused)                              ← AttentionGate inside

        final = fusion(cat([back_prop, forw_prop]))                   ← GatedFusion
        output = upsample(final) + bilinear_base

    ── VRAM estimate vs v1 (n_feats=72, n_blocks=26) ────────────────────────
        TemporalAlignBlocks (2x): ~20 MB
        AttentionGates (26x):     ~8 MB
        GatedFusionBlock extra:   ~2 MB
        Total overhead:           ~30 MB  (P4-safe, plenty of headroom)
    """

    def __init__(self, n_feats=72, n_blocks=26, use_checkpointing=False):
        super().__init__()
        self.n_feats = n_feats
        self.n_blocks = n_blocks

        half_blocks = max(1, n_blocks // 2)

        # 1. Feature Extraction (unchanged)
        self.feat_extract = nn.Conv2d(3, n_feats, 3, 1, 1)

        # 2. Temporal Alignment — aligns neighbor frames to propagated features BEFORE fusion
        self.backward_align = TemporalAlignBlock(n_feats)
        self.forward_align  = TemporalAlignBlock(n_feats)

        # 3. Gated Fusion layers (upgraded from FusionBlock)
        self.backward_fuse = GatedFusionBlock(n_feats * 2, n_feats)
        self.forward_fuse  = GatedFusionBlock(n_feats * 2, n_feats)

        # 4. Propagation Trunks with Attention Gates on skip connections
        self.backward_trunk = nn.ModuleList([
            ResidualBlock(n_feats, use_checkpointing=use_checkpointing, use_attention=True)
            for _ in range(half_blocks)
        ])
        self.forward_trunk = nn.ModuleList([
            ResidualBlock(n_feats, use_checkpointing=use_checkpointing, use_attention=True)
            for _ in range(half_blocks)
        ])

        # 5. Final Gated Fusion
        self.fusion = GatedFusionBlock(n_feats * 2, n_feats)

        # 6. Upsampling 3x (unchanged)
        self.upsample = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * 9, 3, 1, 1),
            nn.PixelShuffle(3),
            nn.Conv2d(n_feats, 3, 3, 1, 1)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor [B, 7, 3, H, W]

        Returns:
            Output tensor [B, 3, H*3, W*3]
        """
        B, T, C, H, W = x.size()

        # Extract features from all 7 frames
        feats = self.feat_extract(x.view(-1, C, H, W))
        feats = feats.view(B, T, self.n_feats, H, W)

        # Initialize propagation from center frame (index 3)
        center_feat = feats[:, 3].clone()

        # ── Backward propagation: center → F5 → F6 → F7 ──────────────────────
        back_prop = center_feat
        for i in [4, 5, 6]:
            # STEP 1: Align neighbor frame to propagated features (motion compensation)
            aligned = self.backward_align(back_prop, feats[:, i])
            # STEP 2: Gated fusion of propagated + aligned neighbor
            fused = self.backward_fuse(torch.cat([back_prop, aligned], dim=1))
            # STEP 3: Process through residual trunk (with attention-gated skip connections)
            for block in self.backward_trunk:
                fused = block(fused)
            back_prop = fused

        # ── Forward propagation: center → F3 → F2 → F1 ───────────────────────
        forw_prop = center_feat
        for i in [2, 1, 0]:
            # STEP 1: Align neighbor frame to propagated features
            aligned = self.forward_align(forw_prop, feats[:, i])
            # STEP 2: Gated fusion of propagated + aligned neighbor
            fused = self.forward_fuse(torch.cat([forw_prop, aligned], dim=1))
            # STEP 3: Process through residual trunk
            for block in self.forward_trunk:
                fused = block(fused)
            forw_prop = fused

        # ── Final fusion of bidirectional features ────────────────────────────
        fused = self.fusion(torch.cat([back_prop, forw_prop], dim=1))

        # ── Upsample with bilinear residual connection ────────────────────────
        base      = F.interpolate(x[:, 3], scale_factor=3, mode='bilinear', align_corners=False)
        upsampled = self.upsample(fused)

        return upsampled + base

    def get_layer_activity(self):
        """
        Returns activity levels for all blocks, including new gates and alignment stats.

        Returns dict with:
            backward_trunk:      list of ResidualBlock activities
            backward_fuse:       [3x3, 1x1, gate] activity
            backward_align_flow: mean flow magnitude from TemporalAlignBlock
            forward_trunk:       list of ResidualBlock activities
            forward_fuse:        [3x3, 1x1, gate] activity
            forward_align_flow:  mean flow magnitude from TemporalAlignBlock
            fusion:              [3x3, 1x1, gate] activity
            attention_gates:     list of dicts {mean, min, max} per ResidualBlock
        """
        backward_activities = [b.last_activity for b in self.backward_trunk]
        forward_activities  = [b.last_activity for b in self.forward_trunk]

        attention_gates = []
        for block in itertools.chain(self.backward_trunk, self.forward_trunk):
            if block.use_attention:
                attention_gates.append({
                    'mean': block.attn_gate.last_gate_mean,
                    'min':  block.attn_gate.last_gate_min,
                    'max':  block.attn_gate.last_gate_max,
                })
            else:
                attention_gates.append(None)

        return {
            'backward_trunk':      backward_activities,
            'backward_fuse':       [self.backward_fuse.last_activity_3x3,
                                    self.backward_fuse.last_activity_1x1,
                                    self.backward_fuse.last_activity_gate],
            'backward_align_flow': self.backward_align.last_flow_magnitude,
            'forward_trunk':       forward_activities,
            'forward_fuse':        [self.forward_fuse.last_activity_3x3,
                                    self.forward_fuse.last_activity_1x1,
                                    self.forward_fuse.last_activity_gate],
            'forward_align_flow':  self.forward_align.last_flow_magnitude,
            'fusion':              [self.fusion.last_activity_3x3,
                                    self.fusion.last_activity_1x1,
                                    self.fusion.last_activity_gate],
            'attention_gates':     attention_gates,
        }

