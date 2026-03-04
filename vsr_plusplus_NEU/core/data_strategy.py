"""
DataStrategyScheduler – Graduated data and loss strategy for progressive training.

Pure Python module (no PyTorch dependency) so it can be tested independently.
"""


class DataStrategyScheduler:
    """
    Graduated data and loss strategy scheduler for progressive training.

    Implements a three-phase training schedule to avoid gradient conflicts
    and premature perceptual loss introduction:

    Phase 1 – Warmup (steps 0–1000):
        Data : 100 % 720_169 (full-frames only)
        Loss : perceptual weight = 0.0

    Phase 2 – Crop Introduction (steps 1000–10000):
        Data : linear interpolation from 100 % full-frames → target distribution
               step 1000  → {720_169: 1.00, 540: 0.00, 720: 0.00}
               step 10000 → {720_169: 0.25, 540: 0.50, 720: 0.25}
        Loss : perceptual weight 0.0 → 0.05 (linear)

    Phase 3 – Stable Training (steps 10000+):
        Data : target distribution (25 % 720_169, 50 % 540, 25 % 720)
        Loss : perceptual weight = 0.05

    Args:
        all_size_keys: list of size keys present in the dataset
                       (sizes absent from TARGET_DISTRIBUTION get weight 0.0)
    """

    PHASE_WARMUP = 'warmup'
    PHASE_CROP_INTRO = 'crop_introduction'
    PHASE_STABLE = 'stable'

    # Phase boundaries (in global training steps)
    WARMUP_END = 1000
    CROP_INTRO_END = 10000

    # Target (final) distribution for Phase 3
    TARGET_DISTRIBUTION = {
        '720_169': 0.25,
        '540': 0.50,
        '720': 0.25,
    }

    # Distribution used during Phase 1 warmup
    WARMUP_DISTRIBUTION = {
        '720_169': 1.0,
        '540': 0.0,
        '720': 0.0,
    }

    # Perceptual weight at end of Phase 2 / throughout Phase 3
    TARGET_PERCEPTUAL_WEIGHT = 0.05

    def __init__(self, all_size_keys=None):
        self._all_size_keys = all_size_keys or list(self.TARGET_DISTRIBUTION.keys())
        self._last_phase = None

    # ------------------------------------------------------------------
    # Phase helpers
    # ------------------------------------------------------------------

    def get_phase(self, step):
        """Return current phase name for the given training step."""
        if step < self.WARMUP_END:
            return self.PHASE_WARMUP
        elif step < self.CROP_INTRO_END:
            return self.PHASE_CROP_INTRO
        else:
            return self.PHASE_STABLE

    # ------------------------------------------------------------------
    # Distribution
    # ------------------------------------------------------------------

    def get_distribution(self, step, available_sizes=None):
        """
        Return per-size sampling weights for the given training step.

        Weights are normalized so they sum to 1.0.  Sizes not in
        TARGET_DISTRIBUTION receive weight 0.0.

        Args:
            step:            Current global training step.
            available_sizes: Iterable of size keys that are actually loaded.
                             Defaults to all_size_keys passed at construction.

        Returns:
            dict mapping size_key → float weight (0.0–1.0, sum == 1.0)
        """
        if available_sizes is None:
            available_sizes = self._all_size_keys

        phase = self.get_phase(step)

        if phase == self.PHASE_WARMUP:
            return {s: self.WARMUP_DISTRIBUTION.get(s, 0.0) for s in available_sizes}

        if phase == self.PHASE_CROP_INTRO:
            t = (step - self.WARMUP_END) / (self.CROP_INTRO_END - self.WARMUP_END)
            t = max(0.0, min(1.0, t))
            dist = {}
            for s in available_sizes:
                start_w = self.WARMUP_DISTRIBUTION.get(s, 0.0)
                end_w = self.TARGET_DISTRIBUTION.get(s, 0.0)
                dist[s] = start_w + t * (end_w - start_w)
            return dist

        # PHASE_STABLE
        return {s: self.TARGET_DISTRIBUTION.get(s, 0.0) for s in available_sizes}

    # ------------------------------------------------------------------
    # Perceptual loss weight
    # ------------------------------------------------------------------

    def get_perceptual_weight(self, step):
        """
        Return the scheduled perceptual loss weight for the given step.

        Returns:
            float in [0.0, TARGET_PERCEPTUAL_WEIGHT]
        """
        phase = self.get_phase(step)

        if phase == self.PHASE_WARMUP:
            return 0.0

        if phase == self.PHASE_CROP_INTRO:
            t = (step - self.WARMUP_END) / (self.CROP_INTRO_END - self.WARMUP_END)
            t = max(0.0, min(1.0, t))
            return t * self.TARGET_PERCEPTUAL_WEIGHT

        return self.TARGET_PERCEPTUAL_WEIGHT

    # ------------------------------------------------------------------
    # Phase-transition logging
    # ------------------------------------------------------------------

    def check_phase_transition(self, step, log_fn=None):
        """
        Detect and optionally log a phase transition.

        Args:
            step:   Current global training step.
            log_fn: Optional callable(message: str) for logging.

        Returns:
            True if a phase transition occurred, False otherwise.
        """
        current_phase = self.get_phase(step)
        if current_phase != self._last_phase:
            if self._last_phase is not None and log_fn is not None:
                log_fn(
                    f"📊 DataStrategy phase transition: "
                    f"{self._last_phase} → {current_phase} at step {step}"
                )
            self._last_phase = current_phase
            return True
        return False
