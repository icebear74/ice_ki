"""
DataStrategyScheduler – Graduated data and loss strategy for progressive training.

Pure Python module (no PyTorch dependency) so it can be tested independently.
"""


class DataStrategyScheduler:
    """
    Graduated data and loss strategy scheduler for progressive training.

    Implements a three-phase training schedule to avoid gradient conflicts
    and premature perceptual loss introduction.

    Key design principle
    -------------------
    The training dataset files on disk are **already pre-weighted** in the
    correct long-run ratio (≈ 50 % 540, 25 % 720, 25 % 720_169).  Therefore
    Phase 3 does NOT apply any distribution override – the sampler simply
    draws proportionally to file counts, which naturally yields the desired
    mix.

    Phase 1 – Warmup (steps 0–15000):
        Data : 100 % 720_169 (full-frames only, via explicit override)
               → model learns global structure without crop conflicts
        Loss : perceptual weight = 0.0

    Phase 2 – Crop Introduction (steps 15000–25000):
        Data : linear interpolation from 100 % 720_169 → approximate natural
               distribution (CROP_INTRO_END_DISTRIBUTION), so that by step
               25000 the override values closely match what is already on disk.
               step 15000 → {720_169: 1.00, 540: 0.00, 720: 0.00}
               step ~25000 → approaching {720_169: 0.25, 540: 0.50, 720: 0.25}
        Loss : perceptual weight 0.0 → 0.05 (linear)

    Phase 3 – Stable Training (steps 25000+):
        Data : natural file-count proportional sampling (no override).
               get_distribution() returns None so the sampler uses the actual
               file ratio on disk (which is already the desired distribution).
        Loss : returns None so the AdaptiveSystem controls the weight
               dynamically (no static override).

    Args:
        all_size_keys: list of size keys present in the dataset
                       (used only during Phase 2 interpolation)
    """

    PHASE_WARMUP = 'warmup'
    PHASE_CROP_INTRO = 'crop_introduction'
    PHASE_STABLE = 'stable'

    # Phase boundaries (in global training steps)
    WARMUP_END = 15000
    CROP_INTRO_END = 25000

    # End-point of the Phase 2 interpolation.
    # This approximates the natural file ratio on disk (50 % 540, 25 % 720,
    # 25 % 720_169).  It is used ONLY during Phase 2; Phase 3 uses None
    # (natural file-count sampling) so the actual on-disk distribution takes
    # over seamlessly.
    CROP_INTRO_END_DISTRIBUTION = {
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
        self._all_size_keys = all_size_keys or list(self.CROP_INTRO_END_DISTRIBUTION.keys())
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

        Phase 1 and 2 return a dict mapping size_key → float weight.
        Phase 3 returns **None**, signalling the sampler to use its default
        file-count proportional mode (no override), because the files on disk
        are already in the desired distribution.

        Args:
            step:            Current global training step.
            available_sizes: Iterable of size keys that are actually loaded.
                             Only used during Phase 1/2.  Defaults to
                             all_size_keys passed at construction.

        Returns:
            dict (Phase 1/2) or None (Phase 3)
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
                end_w = self.CROP_INTRO_END_DISTRIBUTION.get(s, 0.0)
                dist[s] = start_w + t * (end_w - start_w)
            return dist

        # PHASE_STABLE – return None so the sampler uses natural file counts.
        # The files on disk are already in the desired distribution, so no
        # explicit override is needed.
        return None

    # ------------------------------------------------------------------
    # Perceptual loss weight
    # ------------------------------------------------------------------

    def get_perceptual_weight(self, step):
        """
        Return the scheduled perceptual loss weight for the given step.

        Phase 1 (warmup): returns 0.0 to suppress perceptual loss entirely.
        Phase 2 (crop intro): linearly ramps from 0.0 to TARGET_PERCEPTUAL_WEIGHT.
        Phase 3 (stable): returns None so the caller (trainer) keeps the
            AdaptiveSystem's dynamically computed weight instead of overriding it.

        Returns:
            float in [0.0, TARGET_PERCEPTUAL_WEIGHT] for Phase 1/2, or None for Phase 3
        """
        phase = self.get_phase(step)

        if phase == self.PHASE_WARMUP:
            return 0.0

        if phase == self.PHASE_CROP_INTRO:
            t = (step - self.WARMUP_END) / (self.CROP_INTRO_END - self.WARMUP_END)
            t = max(0.0, min(1.0, t))
            return t * self.TARGET_PERCEPTUAL_WEIGHT

        # PHASE_STABLE – return None so the AdaptiveSystem controls the weight.
        return None

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
