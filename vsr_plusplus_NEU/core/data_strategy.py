"""
DataStrategyScheduler – Graduated data and loss strategy for progressive training.

Pure Python module (no PyTorch dependency) so it can be tested independently.

This version is fully dynamic: the warmup template and the Phase-2 end
distribution are derived from the architecture metadata (template pixel areas
and format weights) passed at construction time.  No template name is
hardcoded in this module.

Key design principle
-------------------
The training dataset files on disk are **already pre-weighted** in the
correct long-run ratio (as defined in dataset_architecture.json).  Therefore
Phase 3 does NOT apply any distribution override – the sampler simply draws
proportionally to file counts, which naturally yields the desired mix.

Phase 1 – Warmup (steps 0–3000):
    Data : 100 % warmup template (the one with the LARGEST GT area, so the
           model first learns global structure from the most information-rich
           frames).  Phase 2 will NOT start even at step 3 000 if fewer than
           MIN_CROP_FILES crop files exist (see can_introduce_crops()).
    Loss : perceptual weight = 0.0

Phase 2 – Crop Introduction (steps 3000–8000):
    Data : linear interpolation from 100 % warmup template → end distribution
           (derived from dataset_architecture.json format weights, or from
           equal shares when no weights are available), so that by step 8000
           the override values closely match what is already on disk.
    Loss : perceptual weight 0.0 → TARGET_PERCEPTUAL_WEIGHT (linear)

Phase 3 – Stable Training (steps 8000+):
    Data : natural file-count proportional sampling (no override).
           get_distribution() returns None so the sampler uses the actual
           file ratio on disk (which is already the desired distribution).
    Loss : returns None so the AdaptiveSystem controls the weight
           dynamically (no static override).

Args:
    all_size_keys:    list of template keys present in the dataset
    template_areas:   optional dict mapping template_key → GT pixel count
                      (width * height).  Used to choose the warmup template
                      dynamically.  If None, the first key is used as warmup.
    arch_weights:     optional dict mapping template_key → float weight
                      (from dataset_architecture.json format weights).  Used
                      to build the Phase-2 end distribution.  If None, equal
                      shares are used.
"""


class DataStrategyScheduler:
    """
    Graduated data and loss strategy scheduler for progressive training.
    """

    PHASE_WARMUP = 'warmup'
    PHASE_CROP_INTRO = 'crop_introduction'
    PHASE_STABLE = 'stable'

    # Phase boundaries (in global training steps)
    WARMUP_END = 3000
    CROP_INTRO_END = 8000

    # Minimum number of files that must exist for a given non-warmup template
    # before Phase 2 sampling is allowed to include that template.
    MIN_CROP_FILES = 50

    # Total combined GT images (all non-warmup templates) required before
    # training is allowed to proceed past Phase 1.
    MIN_CROP_FILES_TRAINING = 10000

    # Perceptual weight at end of Phase 2 / throughout Phase 3
    TARGET_PERCEPTUAL_WEIGHT = 0.08

    # Maximum perceptual weight at the end of Phase 1 (late warmup).
    PHASE1_MAX_PERCEPTUAL = 0.03

    def __init__(self, all_size_keys=None, template_areas=None, arch_weights=None):
        all_size_keys = list(all_size_keys) if all_size_keys else []
        self._all_size_keys = all_size_keys

        # ── Choose warmup template ────────────────────────────────────────────
        # Use the template with the largest GT area so the model first learns
        # global structure from the most information-rich format.
        if template_areas and all_size_keys:
            self._warmup_key = max(
                all_size_keys,
                key=lambda k: template_areas.get(k, 0),
            )
        elif all_size_keys:
            self._warmup_key = all_size_keys[0]
        else:
            self._warmup_key = None

        # ── Build Phase-2 end distribution ───────────────────────────────────
        # Derived from arch_weights when available, otherwise equal shares.
        if arch_weights and all_size_keys:
            total_w = sum(arch_weights.get(k, 0.0) for k in all_size_keys)
            if total_w > 0:
                self._end_dist = {k: arch_weights.get(k, 0.0) / total_w for k in all_size_keys}
            else:
                n = len(all_size_keys)
                self._end_dist = {k: 1.0 / n for k in all_size_keys}
        elif all_size_keys:
            n = len(all_size_keys)
            self._end_dist = {k: 1.0 / n for k in all_size_keys}
        else:
            self._end_dist = {}

        # ── Build warmup distribution ─────────────────────────────────────────
        self._warmup_dist = {k: (1.0 if k == self._warmup_key else 0.0)
                             for k in all_size_keys}

        self._last_phase = None

    # ------------------------------------------------------------------
    # Crop-availability guard
    # ------------------------------------------------------------------

    @property
    def CROP_INTRO_END_DISTRIBUTION(self):
        """Phase-2 end distribution (derived from architecture weights)."""
        return dict(self._end_dist)

    @property
    def WARMUP_DISTRIBUTION(self):
        """Phase-1 warmup distribution (100% of warmup template)."""
        return dict(self._warmup_dist)

    @property
    def warmup_template(self):
        """The template key used as the Phase-1 warmup target."""
        return self._warmup_key

    def can_introduce_crops(self, crop_file_counts):
        """
        Return True if there are enough non-warmup files on disk to start Phase 2.

        Args:
            crop_file_counts: dict mapping template_key → int (number of files
                              currently on disk for that template).  Missing keys
                              are treated as 0.

        Returns:
            True if at least one non-warmup template has >= MIN_CROP_FILES files.
        """
        if not crop_file_counts:
            return False
        non_warmup = [k for k in self._all_size_keys if k != self._warmup_key]
        return any(
            crop_file_counts.get(sk, 0) >= self.MIN_CROP_FILES
            for sk in non_warmup
        )

    def get_crop_total_count(self, crop_file_counts):
        """Return the combined file count for all non-warmup templates.

        Args:
            crop_file_counts: dict mapping template_key → int, or None.

        Returns:
            int – total number of non-warmup GT images known to be on disk.
        """
        if not crop_file_counts:
            return 0
        non_warmup = [k for k in self._all_size_keys if k != self._warmup_key]
        return sum(crop_file_counts.get(sk, 0) for sk in non_warmup)

    def has_enough_training_crops(self, crop_file_counts):
        """Return True when combined non-warmup GT images >= MIN_CROP_FILES_TRAINING."""
        return self.get_crop_total_count(crop_file_counts) >= self.MIN_CROP_FILES_TRAINING


    # ------------------------------------------------------------------
    # Phase helpers
    # ------------------------------------------------------------------

    def get_phase(self, step, crop_file_counts=None):
        """Return current phase name for the given training step."""
        if step < self.WARMUP_END:
            return self.PHASE_WARMUP
        if crop_file_counts is not None and not self.can_introduce_crops(crop_file_counts):
            return self.PHASE_WARMUP
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
        Return per-template sampling weights for the given training step.

        Phase 1 and 2 return a dict mapping template_key → float weight.
        Phase 3 returns None, signalling the sampler to use natural file-count
        proportional mode (no override).
        """
        if available_sizes is None:
            available_sizes = self._all_size_keys

        phase = self.get_phase(step, crop_file_counts=crop_file_counts)

        if phase == self.PHASE_WARMUP:
            return {s: self._warmup_dist.get(s, 0.0) for s in available_sizes}

        if phase == self.PHASE_CROP_INTRO:
            t = (step - self.WARMUP_END) / (self.CROP_INTRO_END - self.WARMUP_END)
            t = max(0.0, min(1.0, t))
            dist = {}
            for s in available_sizes:
                start_w = self._warmup_dist.get(s, 0.0)
                end_w = self._end_dist.get(s, 0.0)
                dist[s] = start_w + t * (end_w - start_w)
            return dist

        # PHASE_STABLE – return None so the sampler uses natural file counts.
        return None

    # ------------------------------------------------------------------
    # Perceptual loss weight
    # ------------------------------------------------------------------

    def get_perceptual_weight(self, step, crop_file_counts=None):
        """
        Return the scheduled perceptual loss weight for the given step.

        Phase 1 (warmup): ramps 0.0 → PHASE1_MAX_PERCEPTUAL in the late warmup.
        Phase 2 (crop intro): ramps PHASE1_MAX_PERCEPTUAL → TARGET_PERCEPTUAL_WEIGHT.
        Phase 3 (stable): returns None (AdaptiveSystem controls the weight).
        """
        phase = self.get_phase(step, crop_file_counts=crop_file_counts)

        if phase == self.PHASE_WARMUP:
            if step < 2000:
                return 0.0
            t = (step - 2000) / (self.WARMUP_END - 2000)
            t = max(0.0, min(1.0, t))
            return t * self.PHASE1_MAX_PERCEPTUAL

        if phase == self.PHASE_CROP_INTRO:
            t = (step - self.WARMUP_END) / (self.CROP_INTRO_END - self.WARMUP_END)
            t = max(0.0, min(1.0, t))
            return self.PHASE1_MAX_PERCEPTUAL + t * (
                self.TARGET_PERCEPTUAL_WEIGHT - self.PHASE1_MAX_PERCEPTUAL
            )

        # PHASE_STABLE – return None so the AdaptiveSystem controls the weight.
        return None

    # ------------------------------------------------------------------
    # Phase-transition logging
    # ------------------------------------------------------------------

    def check_phase_transition(self, step, log_fn=None, crop_file_counts=None):
        """
        Detect and optionally log a phase transition.

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
