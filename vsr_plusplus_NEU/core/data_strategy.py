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
    correct long-run ratio (≈ 40 % 720_169, 40 % 720, 20 % 540).  Therefore
    Phase 3 does NOT apply any distribution override – the sampler simply
    draws proportionally to file counts, which naturally yields the desired
    mix.

    Phase 1 – Warmup (steps 0–3000):
        Data : 100 % 720_169 (full-frames only, via explicit override)
               → model learns global structure without crop conflicts.
               Phase 2 will NOT start even at step 3 000 if fewer than
               MIN_CROP_FILES crop files exist (see can_introduce_crops()).
        Loss : perceptual weight = 0.0

    Phase 2 – Crop Introduction (steps 3000–8000):
        Data : linear interpolation from 100 % 720_169 → approximate natural
               distribution (CROP_INTRO_END_DISTRIBUTION), so that by step
               8000 the override values closely match what is already on disk.
               step 3000 → {720_169: 1.00, 720: 0.00, 540: 0.00}
               step ~8000 → approaching {720_169: 0.40, 720: 0.40, 540: 0.20}
        Loss : perceptual weight 0.0 → 0.08 (linear)

    Phase 3 – Stable Training (steps 8000+):
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
    # Phase 1 ends at step 3 000: by then the model has learned basic structure
    # (quality_ki ≈ 0.40) and benefits from crop diversity rather than more
    # full-frame repetition.  Phase 2 will only actually activate once crop
    # files exist (see can_introduce_crops()); until then the sampler keeps
    # the 100 % full-frame distribution regardless of step number.
    WARMUP_END = 3000
    CROP_INTRO_END = 8000

    # Minimum number of files that must exist for a given crop size before
    # Phase 2 sampling is allowed to include that size.
    MIN_CROP_FILES = 50

    # Total combined GT images (540 + 720) required before training is
    # allowed to proceed past the full-frame Phase 1.  Training is paused
    # automatically and checks every 5 minutes until this is met.
    MIN_CROP_FILES_TRAINING = 10000

    # End-point of the Phase 2 interpolation.
    # Matches the actual on-disk distribution (40 % 720_169, 40 % 720,
    # 20 % 540).  It is used ONLY during Phase 2; Phase 3 uses None
    # (natural file-count sampling) so the actual on-disk distribution takes
    # over seamlessly.
    CROP_INTRO_END_DISTRIBUTION = {
        '720_169': 0.40,
        '540': 0.20,
        '720': 0.40,
    }

    # Distribution used during Phase 1 warmup
    WARMUP_DISTRIBUTION = {
        '720_169': 1.0,
        '540': 0.0,
        '720': 0.0,
    }

    # Perceptual weight at end of Phase 2 / throughout Phase 3
    TARGET_PERCEPTUAL_WEIGHT = 0.08

    # Maximum perceptual weight at the end of Phase 1 (late warmup).
    # Steps 0–2000: no perceptual (let basic structure stabilize).
    # Steps 2000–WARMUP_END: gentle linear ramp 0.0 → PHASE1_MAX_PERCEPTUAL.
    # Phase 2 then continues the ramp PHASE1_MAX_PERCEPTUAL → TARGET_PERCEPTUAL_WEIGHT.
    PHASE1_MAX_PERCEPTUAL = 0.03

    def __init__(self, all_size_keys=None):
        self._all_size_keys = all_size_keys or list(self.CROP_INTRO_END_DISTRIBUTION.keys())
        self._last_phase = None

    # ------------------------------------------------------------------
    # Crop-availability guard
    # ------------------------------------------------------------------

    @classmethod
    def can_introduce_crops(cls, crop_file_counts):
        """
        Return True if there are enough crop files on disk to start Phase 2.

        Crop files are generated externally (from 4K source material) and may
        not be available at the step when WARMUP_END is reached.  This guard
        prevents the sampler from requesting batches from empty size-groups.

        Args:
            crop_file_counts: dict mapping size_key → int (number of files
                              currently on disk for that size).  Keys '540'
                              and '720' are the crop sizes.  Missing keys are
                              treated as 0.

        Returns:
            True if at least one crop size has >= MIN_CROP_FILES files.
        """
        if not crop_file_counts:
            return False
        crop_sizes = ('540', '720')
        return any(
            crop_file_counts.get(sk, 0) >= cls.MIN_CROP_FILES
            for sk in crop_sizes
        )

    @classmethod
    def get_crop_total_count(cls, crop_file_counts):
        """Return the combined file count for all crop sizes (540 + 720).

        Args:
            crop_file_counts: dict mapping size_key → int, or None.

        Returns:
            int – total number of crop GT images known to be on disk.
        """
        if not crop_file_counts:
            return 0
        return sum(crop_file_counts.get(sk, 0) for sk in ('540', '720'))

    @classmethod
    def has_enough_training_crops(cls, crop_file_counts):
        """Return True when combined 540+720 GT images >= MIN_CROP_FILES_TRAINING.

        This is the *training-pause* guard: training is blocked at WARMUP_END
        until this stricter threshold is satisfied, ensuring Phase 2 has
        sufficient crop diversity before it starts.

        Args:
            crop_file_counts: dict mapping size_key → int, or None.

        Returns:
            bool
        """
        return cls.get_crop_total_count(crop_file_counts) >= cls.MIN_CROP_FILES_TRAINING

    # ------------------------------------------------------------------
    # Phase helpers
    # ------------------------------------------------------------------

    def get_phase(self, step, crop_file_counts=None):
        """Return current phase name for the given training step.

        If *crop_file_counts* is provided and Phase 2 would normally start
        (step >= WARMUP_END) but not enough crop files exist yet, the phase
        stays 'warmup' until crops are available.
        """
        if step < self.WARMUP_END:
            return self.PHASE_WARMUP
        # Phase 2 / 3 are only entered when crop files actually exist.
        if crop_file_counts is not None and not self.can_introduce_crops(crop_file_counts):
            return self.PHASE_WARMUP
        # If no crop counts are available and we have passed WARMUP_END, warn
        # once so integration issues are visible without crashing training.
        if crop_file_counts is None and step >= self.WARMUP_END:
            import warnings
            warnings.warn(
                f"DataStrategyScheduler.get_phase() called at step {step} "
                f"(>= WARMUP_END={self.WARMUP_END}) without crop_file_counts. "
                "Phase 2/3 will proceed based on step count alone; pass "
                "crop_file_counts to guard against empty crop directories.",
                stacklevel=2,
            )
        if step < self.CROP_INTRO_END:
            return self.PHASE_CROP_INTRO
        return self.PHASE_STABLE

    # ------------------------------------------------------------------
    # Distribution
    # ------------------------------------------------------------------

    def get_distribution(self, step, available_sizes=None, crop_file_counts=None):
        """
        Return per-size sampling weights for the given training step.

        Phase 1 and 2 return a dict mapping size_key → float weight.
        Phase 3 returns **None**, signalling the sampler to use its default
        file-count proportional mode (no override), because the files on disk
        are already in the desired distribution.

        Args:
            step:             Current global training step.
            available_sizes:  Iterable of size keys that are actually loaded.
                              Only used during Phase 1/2.  Defaults to
                              all_size_keys passed at construction.
            crop_file_counts: Optional dict mapping size_key → int of files
                              currently on disk.  When provided, Phase 2 is
                              held back until enough crop files exist (see
                              can_introduce_crops()).

        Returns:
            dict (Phase 1/2) or None (Phase 3)
        """
        if available_sizes is None:
            available_sizes = self._all_size_keys

        phase = self.get_phase(step, crop_file_counts=crop_file_counts)

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

    def get_perceptual_weight(self, step, crop_file_counts=None):
        """
        Return the scheduled perceptual loss weight for the given step.

        Phase 1 (warmup): returns 0.0 to suppress perceptual loss entirely.
        Phase 2 (crop intro): linearly ramps from 0.0 to TARGET_PERCEPTUAL_WEIGHT.
        Phase 3 (stable): returns None so the caller (trainer) keeps the
            AdaptiveSystem's dynamically computed weight instead of overriding it.

        Args:
            step:             Current global training step.
            crop_file_counts: Optional dict mapping size_key → int; forwarded
                              to get_phase() to honour the crop-existence guard.

        Returns:
            float in [0.0, TARGET_PERCEPTUAL_WEIGHT] for Phase 1/2, or None for Phase 3
        """
        phase = self.get_phase(step, crop_file_counts=crop_file_counts)

        if phase == self.PHASE_WARMUP:
            # Early warmup (steps 0–2000): no perceptual — let structure stabilize first.
            if step < 2000:
                return 0.0
            # Late warmup (steps 2000–WARMUP_END): gentle ramp 0.0 → PHASE1_MAX_PERCEPTUAL
            # so the model gets early perceptual feedback without destabilising structure.
            t = (step - 2000) / (self.WARMUP_END - 2000)
            t = max(0.0, min(1.0, t))
            return t * self.PHASE1_MAX_PERCEPTUAL

        if phase == self.PHASE_CROP_INTRO:
            t = (step - self.WARMUP_END) / (self.CROP_INTRO_END - self.WARMUP_END)
            t = max(0.0, min(1.0, t))
            # Ramp from PHASE1_MAX_PERCEPTUAL → TARGET_PERCEPTUAL_WEIGHT for a smooth
            # transition (Phase 1 ends at PHASE1_MAX_PERCEPTUAL, not at 0.0).
            return self.PHASE1_MAX_PERCEPTUAL + t * (self.TARGET_PERCEPTUAL_WEIGHT - self.PHASE1_MAX_PERCEPTUAL)

        # PHASE_STABLE – return None so the AdaptiveSystem controls the weight.
        return None

    # ------------------------------------------------------------------
    # Phase-transition logging
    # ------------------------------------------------------------------

    def check_phase_transition(self, step, log_fn=None, crop_file_counts=None):
        """
        Detect and optionally log a phase transition.

        Args:
            step:             Current global training step.
            log_fn:           Optional callable(message: str) for logging.
            crop_file_counts: Optional dict mapping size_key → int; forwarded
                              to get_phase() to honour the crop-existence guard.

        Returns:
            True if a phase transition occurred, False otherwise.
        """
        current_phase = self.get_phase(step, crop_file_counts=crop_file_counts)
        if current_phase != self._last_phase:
            if self._last_phase is not None and log_fn is not None:
                log_fn(
                    f"📊 DataStrategy phase transition: "
                    f"{self._last_phase} → {current_phase} at step {step}"
                )
            self._last_phase = current_phase
            return True
        return False
