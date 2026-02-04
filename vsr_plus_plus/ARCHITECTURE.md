# VSR++ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          VSR++ Training System                          │
│                     Next-Gen Video Super-Resolution                     │
└─────────────────────────────────────────────────────────────────────────┘

                              train.py (Entry Point)
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
              User Choice                         System Init
                    │                                   │
        ┌───────────┴──────────┐              ┌────────┴────────┐
        │                      │              │                 │
   [L]öschen            [F]ortsetzen    Model Creation    Data Loading
        │                      │              │                 │
  Auto-Tune              Load Config    Optimizer/LR      Train/Val Sets
        │                      │              │                 │
        └──────────────────────┴──────────────┴─────────────────┘
                                      │
                            ┌─────────┴─────────┐
                            │   VSRTrainer      │
                            │  (Main Loop)      │
                            └─────────┬─────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                 │
              Training Loop     Validation        Checkpointing
                    │                 │                 │
        ┌───────────┴────────┐        │        ┌────────┴────────┐
        │                    │        │        │                 │
   Model Forward      Adaptive        │    Regular        Best
        │             System          │    (10k steps)  (2k-8k window)
        │                 │           │        │                 │
   Hybrid Loss      Loss Weights      │    .pth files      Symlinks
        │            Grad Clip         │                         │
   Backward              │             │              checkpoint_best.pth
        │                 │            │              checkpoint_best_old.pth
   Optimizer        LR Schedule        │
        │                 │            │
        └─────────────────┴────────────┘
                          │
                    ┌─────┴─────┐
                    │           │
              File Logging   TensorBoard
                    │           │
            training.log    17+ Graphs
        training_status.txt


┌─────────────────────────────────────────────────────────────────────────┐
│                           Component Details                              │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────┐
│  VSRBidirectional_3x     │
│  (Core Model)            │
├──────────────────────────┤
│ Input:  [B,5,3,180,180] │
│ Output: [B,3,540,540]   │
│                          │
│ 1. Feature Extract      │
│ 2. Frame-3 Init         │
│    center_feat = F[2]   │
│ 3. Backward: F3→F4→F5   │
│ 4. Forward:  F3→F2→F1   │
│ 5. Fusion               │
│ 6. Upsample 3x          │
└──────────────────────────┘

┌──────────────────────────┐
│  HybridLoss              │
│  (Loss Function)         │
├──────────────────────────┤
│ • L1 Loss               │
│ • Multi-Scale Loss      │
│ • Gradient Loss         │
│                          │
│ Total = w₁·L1 +         │
│         w₂·MS +         │
│         w₃·Grad         │
└──────────────────────────┘

┌──────────────────────────┐
│  Auto-Tune System        │
│  (Configuration)         │
├──────────────────────────┤
│ Test 8 Configs:         │
│ 1. n_feats=192, b=3     │
│ 2. n_feats=160, b=4     │
│ ...                      │
│ 8. n_feats=64, b=4      │
│                          │
│ Criteria:               │
│ • Speed ≤ 4.0s/iter     │
│ • VRAM ≤ 80% max        │
│ • Batch ≥ 4 (effective) │
└──────────────────────────┘

┌──────────────────────────┐
│  Checkpoint Manager      │
│  (Smart Saving)          │
├──────────────────────────┤
│ Regular: Every 10k      │
│   checkpoint_step_*.pth │
│                          │
│ Best: During 2k-8k      │
│   checkpoint_best.pth → │
│   checkpoint_step_*.pth │
│                          │
│ Emergency: On crash     │
│   checkpoint_emergency  │
└──────────────────────────┘

┌──────────────────────────┐
│  Adaptive System         │
│  (Dynamic Training)      │
├──────────────────────────┤
│ Loss Weights:           │
│ • Normal Mode           │
│   Adjust every 50 steps │
│ • Aggressive Mode       │
│   Adjust every 10 steps │
│   Boost Grad to 0.30    │
│                          │
│ Gradient Clipping:      │
│ • Track last 500 norms  │
│ • Clip at 95th %ile     │
│                          │
│ LR Scheduling:          │
│ • Warmup: 0 → 1e-4      │
│ • Cosine: 1e-4 → 1e-6   │
│ • Plateau: LR × 0.5     │
└──────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────┐
│                        TensorBoard Graphs (17+)                          │
└─────────────────────────────────────────────────────────────────────────┘

Loss/          │ L1, MS, Grad, Total
Training/      │ LearningRate
Adaptive/      │ L1_Weight, MS_Weight, Grad_Weight, GradientClip, AggressiveMode
Quality/       │ LR_Quality, KI_Quality, Improvement
Metrics/       │ LR_PSNR, LR_SSIM, KI_PSNR, KI_SSIM
System/        │ VRAM_GB, Speed_s_per_iter
Gradients/     │ TotalNorm
Activity/      │ Backward_Trunk_Avg, Forward_Trunk_Avg
LR_Schedule/   │ Phase
Events/        │ Checkpoints
Validation/    │ Comparison (images)


┌─────────────────────────────────────────────────────────────────────────┐
│                            Data Flow                                     │
└─────────────────────────────────────────────────────────────────────────┘

Dataset                Model              Validator           Logger
   │                     │                    │                  │
   ├─► LR: 180x900 ────►│                    │                  │
   │   (5 frames)       │                    │                  │
   │                    │                    │                  │
   ├─► GT: 540x540 ────►│                    │                  │
   │                    │                    │                  │
   │              ┌─────┴─────┐              │                  │
   │              │  Forward  │              │                  │
   │              │   Pass    │              │                  │
   │              └─────┬─────┘              │                  │
   │                    │                    │                  │
   │              Output: 540x540            │                  │
   │                    │                    │                  │
   └────────────────────┴────────────────────┤                  │
                                            │                  │
                                      Quality Metrics          │
                                            │                  │
                                            ├─────────────────►│
                                            │           Log to files
                                            │           Log to TB
                                            │                  │
                                            └──────────────────┘


┌─────────────────────────────────────────────────────────────────────────┐
│                         Usage Flow                                       │
└─────────────────────────────────────────────────────────────────────────┘

1. Start: python vsr_plus_plus/train.py

2. Choice:
   [L] Delete → Auto-tune → Find optimal config → Start fresh
   [F] Resume → Load config → Load checkpoint → Continue training

3. Training:
   Every step:      Forward → Loss → Backward → Update
   Every 5 steps:   Update status file
   Every 100 steps: Log to TensorBoard
   Every 500 steps: Run validation
   Every 10k steps: Save regular checkpoint
   When best:       Save best checkpoint + update symlinks

4. Monitor:
   tensorboard --logdir .../logs --bind_all
   Open http://localhost:6006

5. Results:
   Checkpoints in /mnt/data/training/.../Learn/
   Logs in /mnt/data/training/.../Learn/logs/
   Status in training_status.txt
   Events in training.log
```
