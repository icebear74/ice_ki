"""ComfyUI workflow graph analysis module.

Analyzes ComfyUI workflow JSON (prompt-API format, i.e. a flat dict of
``{node_id: {class_type, inputs}}`` records) and identifies key graph roles
without relying on fixed node IDs.

The analysis result is a :class:`WorkflowRoles` instance that carries:

- lists of node IDs by type (samplers, model loaders, CLIP encoders, …)
- *resolved* primary roles (primary sampler, model loader, positive/negative
  CLIPTextEncode nodes to inject into, latent-image node, …)
- flags for img2img-relevant structures and negative ConditioningZeroOut
- validation state: ``is_usable``, ``warnings``, ``errors``

Design goals
------------
* No fixed-ID assumptions – works with any ComfyUI-exported workflow.
* Handles UNET/DiffusionModelLoader workflows, dual-CLIP (FLUX), ZeroOut
  negative conditioning, multi-sampler pipelines.
* Prepared for future img2img: detects VAEEncode + image-input paths.
* Returns clear diagnostics when the graph is ambiguous.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Node-type classification sets
# ---------------------------------------------------------------------------

SAMPLER_TYPES: frozenset[str] = frozenset({
    "KSampler",
    "KSamplerAdvanced",
    "SamplerCustomAdvanced",  # FLUX / UNet-based workflows
})

CHECKPOINT_LOADER_TYPES: frozenset[str] = frozenset({
    "CheckpointLoaderSimple",
    "CheckpointLoader",
})

UNET_LOADER_TYPES: frozenset[str] = frozenset({
    "UNETLoader",
    "DiffusionModelLoader",
})

MODEL_LOADER_TYPES: frozenset[str] = CHECKPOINT_LOADER_TYPES | UNET_LOADER_TYPES

# Nodes that modify / merge / wrap a model but pass it through – we look
# upstream through these to find the actual loader.
MODEL_PASSTHROUGH_TYPES: frozenset[str] = frozenset({
    "LoraLoader",
    "LoraLoaderModelOnly",
    "ModelMergeSimple",
    "ModelMergeBlocks",
    "ModelSamplingFlux",
    "ModelSamplingAuraFlow",
    "ModelSamplingDiscrete",
    "FreeU",
    "FreeU_V2",
    "PatchModelAddDownscale",
    "LatentConsistencyModelMerge",
    "FluxGuidance",
})

# Guider nodes used by SamplerCustomAdvanced workflows (FLUX / UNet-based).
# These carry the model + conditioning references that would normally be on the
# KSampler node directly.
GUIDER_TYPES: frozenset[str] = frozenset({
    "BasicGuider",
    "CFGGuider",
    "DualCFGGuider",
    "PAGGuider",
})

# Conditioning modifiers that pass conditioning through – we look upstream
# through these to find CLIPTextEncode leaf nodes.
CONDITIONING_PASSTHROUGH_TYPES: frozenset[str] = frozenset({
    "ConditioningCombine",
    "ConditioningConcat",
    "ConditioningAverage",
    "ConditioningSetArea",
    "ConditioningSetAreaPercentage",
    "ConditioningSetAreaStrength",
    "ConditioningSetMask",
    "ConditioningSetMaskAndCombine",
    "ConditioningSetTimestepRange",
    "ConditioningMultiply",
    "ConditioningStableAudio",
    # CLIP wrappers that still carry the conditioning through
    "CLIPSetLastLayer",
})

LATENT_IMAGE_TYPES: frozenset[str] = frozenset({
    "EmptyLatentImage",
    "EmptySD3LatentImage",
    "EmptyHunyuanLatentVideo",
    "EmptyMochiLatentVideo",
    "EmptyCogVideoXLatentVideo",
})

VAE_DECODE_TYPES: frozenset[str] = frozenset({
    "VAEDecode",
    "VAEDecodeTiled",
})

VAE_ENCODE_TYPES: frozenset[str] = frozenset({
    "VAEEncode",
    "VAEEncodeTiled",
    "VAEEncodeForInpaint",
})

IMAGE_INPUT_TYPES: frozenset[str] = frozenset({
    "LoadImage",
    "ImageBatch",
    "LoadImageMask",
})

OUTPUT_TYPES: frozenset[str] = frozenset({
    "SaveImage",
    "PreviewImage",
})


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class WorkflowRoles:
    """Structured result of analyzing a ComfyUI workflow graph.

    All ``*_ids`` lists preserve the order in which the nodes appear in the
    workflow dict.  Primary IDs point to the single "best" node for each role.
    ``is_usable`` is ``True`` only when a usable sampler was found and no
    hard errors were detected.
    """

    __slots__ = (
        # ── lists of node IDs by type ──────────────────────────────────────
        "sampler_ids",
        "checkpoint_loader_ids",
        "unet_loader_ids",
        "model_loader_ids",
        "clip_text_encode_ids",
        "latent_image_ids",
        "vae_decode_ids",
        "vae_encode_ids",
        "image_input_ids",
        "output_ids",
        # ── resolved primary roles ─────────────────────────────────────────
        "primary_sampler_id",
        "primary_model_loader_id",
        "positive_clip_ids",   # all CLIPTextEncode IDs feeding the positive input
        "negative_clip_ids",   # all CLIPTextEncode IDs feeding the negative input
        "primary_latent_id",
        # ── flags ─────────────────────────────────────────────────────────
        "negative_is_zero_out",
        "is_potentially_img2img",
        "model_type",          # "checkpoint" | "unet" | "any"
        # ── validation ────────────────────────────────────────────────────
        "is_usable",
        "warnings",
        "errors",
    )

    def __init__(self) -> None:
        self.sampler_ids: list[str] = []
        self.checkpoint_loader_ids: list[str] = []
        self.unet_loader_ids: list[str] = []
        self.model_loader_ids: list[str] = []
        self.clip_text_encode_ids: list[str] = []
        self.latent_image_ids: list[str] = []
        self.vae_decode_ids: list[str] = []
        self.vae_encode_ids: list[str] = []
        self.image_input_ids: list[str] = []
        self.output_ids: list[str] = []

        self.primary_sampler_id: str | None = None
        self.primary_model_loader_id: str | None = None
        self.positive_clip_ids: list[str] = []
        self.negative_clip_ids: list[str] = []
        self.primary_latent_id: str | None = None

        self.negative_is_zero_out: bool = False
        self.is_potentially_img2img: bool = False
        self.model_type: str = "any"

        self.is_usable: bool = False
        self.warnings: list[str] = []
        self.errors: list[str] = []

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary suitable for template metadata."""
        return {
            "is_usable": self.is_usable,
            "model_type": self.model_type,
            "sampler_count": len(self.sampler_ids),
            "model_loader_count": len(self.model_loader_ids),
            "positive_clip_count": len(self.positive_clip_ids),
            "negative_clip_count": len(self.negative_clip_ids),
            "latent_image_count": len(self.latent_image_ids),
            "negative_is_zero_out": self.negative_is_zero_out,
            "is_potentially_img2img": self.is_potentially_img2img,
            "primary_sampler_id": self.primary_sampler_id,
            "primary_model_loader_id": self.primary_model_loader_id,
            "positive_clip_ids": self.positive_clip_ids,
            "negative_clip_ids": self.negative_clip_ids,
            "primary_latent_id": self.primary_latent_id,
            "warnings": self.warnings,
            "errors": self.errors,
        }


# ---------------------------------------------------------------------------
# Internal graph traversal helpers
# ---------------------------------------------------------------------------

def _resolve_node(workflow: dict[str, Any], ref: Any) -> tuple[str, dict[str, Any]] | tuple[None, None]:
    """Resolve a ComfyUI ``[node_id, slot]`` reference to ``(id, node)``."""
    if isinstance(ref, list) and len(ref) >= 1:
        nid = str(ref[0])
        node = workflow.get(nid)
        if isinstance(node, dict):
            return nid, node
    return None, None


def _find_output_sampler(
    workflow: dict[str, Any],
    sampler_ids: list[str],
) -> str | None:
    """Return the sampler whose output is consumed by a VAEDecode node."""
    vae_input_ids: set[str] = set()
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        if node.get("class_type") in VAE_DECODE_TYPES:
            nid, _ = _resolve_node(workflow, node.get("inputs", {}).get("samples"))
            if nid:
                vae_input_ids.add(nid)
    for sid in sampler_ids:
        if sid in vae_input_ids:
            return sid
    return None


def _find_all_clips_in_conditioning_chain(
    workflow: dict[str, Any],
    node_id: str,
    visited: frozenset[str] | None = None,
    depth: int = 0,
) -> tuple[list[str], bool]:
    """Recursively find all CLIPTextEncode nodes feeding a conditioning chain.

    Returns ``(clip_ids, is_zero_out)``.  ``is_zero_out`` is ``True`` when the
    chain ends in a ``ConditioningZeroOut`` node (no injection possible).
    """
    if visited is None:
        visited = frozenset()
    if node_id in visited or depth > 16:
        return [], False
    visited = visited | {node_id}

    node = workflow.get(node_id)
    if not isinstance(node, dict):
        return [], False

    ct = node.get("class_type", "")

    if ct == "CLIPTextEncode":
        return [node_id], False

    if ct == "ConditioningZeroOut":
        return [], True

    inputs = node.get("inputs", {})

    # For passthrough conditioning nodes, search all conditioning inputs
    all_clips: list[str] = []
    for key in (
        "conditioning_1",
        "conditioning_2",
        "conditioning",
        "base_conditioning",
        "conditioning_to",
        "conditioning_from",
    ):
        ref = inputs.get(key)
        if not isinstance(ref, list):
            continue
        ref_id = str(ref[0]) if ref else None
        if not ref_id or ref_id in visited:
            continue
        clips, is_zero = _find_all_clips_in_conditioning_chain(workflow, ref_id, visited, depth + 1)
        if is_zero:
            return [], True  # ZeroOut propagates upward
        all_clips.extend(clips)

    return all_clips, False


def _trace_to_model_loader(
    workflow: dict[str, Any],
    node_id: str,
    visited: frozenset[str] | None = None,
    depth: int = 0,
) -> str | None:
    """Follow model-modifying nodes upstream to find the actual loader node."""
    if visited is None:
        visited = frozenset()
    if node_id in visited or depth > 12:
        return None
    visited = visited | {node_id}

    node = workflow.get(node_id)
    if not isinstance(node, dict):
        return None

    ct = node.get("class_type", "")
    if ct in MODEL_LOADER_TYPES:
        return node_id

    inputs = node.get("inputs", {})
    for key in ("model_1", "model", "model1", "unet"):
        ref = inputs.get(key)
        if not isinstance(ref, list):
            continue
        ref_id = str(ref[0]) if ref else None
        if ref_id and ref_id not in visited:
            result = _trace_to_model_loader(workflow, ref_id, visited, depth + 1)
            if result is not None:
                return result
    return None


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def analyze_workflow(workflow: dict[str, Any]) -> WorkflowRoles:
    """Analyze a ComfyUI workflow dict and return a :class:`WorkflowRoles` result.

    Parameters
    ----------
    workflow:
        Flat ``{node_id: {class_type, inputs, …}}`` dict as used by ComfyUI's
        ``POST /prompt`` API.  Extra keys like ``_meta`` are ignored.

    Returns
    -------
    WorkflowRoles
        See class docstring.  ``is_usable`` is the quick-check; inspect
        ``errors`` and ``warnings`` for details.
    """
    roles = WorkflowRoles()

    if not isinstance(workflow, dict):
        roles.errors.append("Workflow ist kein JSON-Objekt.")
        return roles

    # ── 1. Classify every node ───────────────────────────────────────────────
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        ct = node.get("class_type", "")
        if ct in SAMPLER_TYPES:
            roles.sampler_ids.append(node_id)
        if ct in CHECKPOINT_LOADER_TYPES:
            roles.checkpoint_loader_ids.append(node_id)
            roles.model_loader_ids.append(node_id)
        if ct in UNET_LOADER_TYPES:
            roles.unet_loader_ids.append(node_id)
            roles.model_loader_ids.append(node_id)
        if ct == "CLIPTextEncode":
            roles.clip_text_encode_ids.append(node_id)
        if ct in LATENT_IMAGE_TYPES:
            roles.latent_image_ids.append(node_id)
        if ct in VAE_DECODE_TYPES:
            roles.vae_decode_ids.append(node_id)
        if ct in VAE_ENCODE_TYPES:
            roles.vae_encode_ids.append(node_id)
        if ct in IMAGE_INPUT_TYPES:
            roles.image_input_ids.append(node_id)
        if ct in OUTPUT_TYPES:
            roles.output_ids.append(node_id)

    # ── 2. Determine primary sampler ─────────────────────────────────────────
    if not roles.sampler_ids:
        roles.errors.append(
            "Kein KSampler/KSamplerAdvanced-Knoten gefunden. "
            "Das Workflow-JSON muss einen unterstützten Sampler-Knoten enthalten."
        )
        return roles  # nothing more we can determine

    if len(roles.sampler_ids) == 1:
        roles.primary_sampler_id = roles.sampler_ids[0]
    else:
        # Multiple samplers: prefer the one whose output feeds VAEDecode
        output_sampler = _find_output_sampler(workflow, roles.sampler_ids)
        if output_sampler is not None:
            roles.primary_sampler_id = output_sampler
            roles.warnings.append(
                f"{len(roles.sampler_ids)} Sampler gefunden; "
                "verwende den, der an VAEDecode angeschlossen ist "
                f"(Knoten-ID {output_sampler!r})."
            )
        else:
            roles.primary_sampler_id = roles.sampler_ids[0]
            roles.warnings.append(
                f"{len(roles.sampler_ids)} Sampler gefunden, "
                "aber kein eindeutiger Ausgabe-Sampler erkennbar; "
                f"verwende ersten (Knoten-ID {roles.sampler_ids[0]!r})."
            )

    sampler_node = workflow[roles.primary_sampler_id]
    sampler_inputs = sampler_node.get("inputs", {})

    # ── 2b. For SamplerCustomAdvanced: resolve inputs through the guider node ──
    # SamplerCustomAdvanced does not have direct positive/negative/model inputs.
    # Instead, they live on the connected guider (CFGGuider, BasicGuider, …).
    _guider_inputs: dict[str, Any] = {}
    if sampler_node.get("class_type") == "SamplerCustomAdvanced":
        guider_ref = sampler_inputs.get("guider")
        if isinstance(guider_ref, list) and guider_ref:
            guider_id = str(guider_ref[0])
            guider_node = workflow.get(guider_id)
            if isinstance(guider_node, dict) and guider_node.get("class_type") in GUIDER_TYPES:
                _guider_inputs = guider_node.get("inputs", {})

    # ── 3. Positive conditioning chain ───────────────────────────────────────
    # For KSampler/KSamplerAdvanced: direct "positive" input on the sampler.
    # For SamplerCustomAdvanced:    "positive" or "conditioning" on the guider.
    pos_ref = (
        _guider_inputs.get("positive") or _guider_inputs.get("conditioning")
        if _guider_inputs
        else sampler_inputs.get("positive")
    )
    if isinstance(pos_ref, list) and pos_ref:
        pos_id = str(pos_ref[0])
        clip_ids, is_zero = _find_all_clips_in_conditioning_chain(workflow, pos_id)
        if clip_ids:
            roles.positive_clip_ids = clip_ids
            if len(clip_ids) > 1:
                roles.warnings.append(
                    f"Positives Conditioning enthält {len(clip_ids)} CLIPTextEncode-Knoten "
                    f"(IDs: {clip_ids}) – Prompt wird in alle injiziert (z. B. duales CLIP für FLUX)."
                )
        elif is_zero:
            roles.warnings.append(
                "Positives Conditioning endet in ConditioningZeroOut – kein Prompt injizierbar."
            )
        else:
            roles.warnings.append(
                "Positives Conditioning: kein injizierbares CLIPTextEncode gefunden "
                "(z. B. komplexer Conditioning-Graph). Prompt kann nicht überschrieben werden."
            )
    else:
        roles.warnings.append("Sampler hat keinen 'positive'-Eingang – positiver Prompt kann nicht gesetzt werden.")

    # ── 4. Negative conditioning chain ───────────────────────────────────────
    # For KSampler/KSamplerAdvanced: direct "negative" input.
    # For SamplerCustomAdvanced:    "negative" on the guider (only CFGGuider has one).
    neg_ref = (
        _guider_inputs.get("negative")
        if _guider_inputs
        else sampler_inputs.get("negative")
    )
    if isinstance(neg_ref, list) and neg_ref:
        neg_id = str(neg_ref[0])
        neg_node = workflow.get(neg_id)
        if isinstance(neg_node, dict) and neg_node.get("class_type") == "ConditioningZeroOut":
            # Intentionally zeroed-out negative (common in FLUX/AuraFlow)
            roles.negative_is_zero_out = True
        else:
            clip_ids, is_zero = _find_all_clips_in_conditioning_chain(workflow, neg_id)
            if clip_ids:
                roles.negative_clip_ids = clip_ids
                if len(clip_ids) > 1:
                    roles.warnings.append(
                        f"Negatives Conditioning enthält {len(clip_ids)} CLIPTextEncode-Knoten – "
                        "Negativ-Prompt wird in alle injiziert."
                    )
            elif is_zero:
                roles.negative_is_zero_out = True
            # else: no injectable negative – silently skip, no warning needed

    # ── 5. Latent image / img2img detection ──────────────────────────────────
    latent_ref = sampler_inputs.get("latent_image")
    if isinstance(latent_ref, list) and latent_ref:
        latent_id = str(latent_ref[0])
        latent_node = workflow.get(latent_id)
        if isinstance(latent_node, dict):
            latent_ct = latent_node.get("class_type", "")
            if latent_ct in LATENT_IMAGE_TYPES:
                roles.primary_latent_id = latent_id
            elif latent_ct in VAE_ENCODE_TYPES:
                roles.is_potentially_img2img = True
                roles.warnings.append(
                    f"Latent-Quelle ist '{latent_ct}' (Knoten {latent_id!r}) – "
                    "das deutet auf einen img2img-Workflow hin. "
                    "Die WebUI unterstützt img2img derzeit noch nicht vollständig."
                )
    if roles.primary_latent_id is None and roles.latent_image_ids:
        # Fallback: use first EmptyLatentImage-type node in workflow
        roles.primary_latent_id = roles.latent_image_ids[0]
        if not latent_ref:
            roles.warnings.append(
                f"Sampler hat keinen direkten 'latent_image'-Eingang; "
                f"verwende ersten Latent-Knoten (ID {roles.latent_image_ids[0]!r})."
            )

    # ── 6. Image-input / img2img paths ───────────────────────────────────────
    if roles.vae_encode_ids or roles.image_input_ids:
        roles.is_potentially_img2img = True

    # ── 7. Model loader ───────────────────────────────────────────────────────
    # For KSampler/KSamplerAdvanced: direct "model" input.
    # For SamplerCustomAdvanced:    "model" on the guider node.
    model_ref = (
        _guider_inputs.get("model")
        if _guider_inputs
        else sampler_inputs.get("model")
    )
    if isinstance(model_ref, list) and model_ref:
        model_id = str(model_ref[0])
        loader_id = _trace_to_model_loader(workflow, model_id)
        if loader_id:
            roles.primary_model_loader_id = loader_id
        elif model_id in workflow:
            # The model input points to something, but we couldn't trace it to a loader
            roles.warnings.append(
                f"Modell-Eingang des Samplers zeigt auf Knoten {model_id!r} "
                f"({workflow[model_id].get('class_type', '?')}), aber kein Loader-Knoten erreichbar. "
                "Modellname kann evtl. nicht gesetzt werden."
            )

    if roles.primary_model_loader_id is None and roles.model_loader_ids:
        # Fallback: use first loader found anywhere in the workflow
        roles.primary_model_loader_id = roles.model_loader_ids[0]
        if len(roles.model_loader_ids) > 1:
            roles.warnings.append(
                f"{len(roles.model_loader_ids)} Modell-Loader gefunden; "
                f"verwende ersten (Knoten-ID {roles.model_loader_ids[0]!r})."
            )

    if not roles.model_loader_ids:
        roles.warnings.append(
            "Kein Modell-Loader-Knoten gefunden "
            "(CheckpointLoaderSimple, UNETLoader, DiffusionModelLoader). "
            "Modellname kann nicht gesetzt werden."
        )

    # ── 8. Model type ─────────────────────────────────────────────────────────
    if roles.unet_loader_ids:
        roles.model_type = "unet"
    elif roles.checkpoint_loader_ids:
        roles.model_type = "checkpoint"
    else:
        roles.model_type = "any"

    # ── 9. Final usability judgement ─────────────────────────────────────────
    roles.is_usable = len(roles.errors) == 0 and roles.primary_sampler_id is not None

    logger.debug(
        "workflow_analyzer: usable=%s model_type=%s samplers=%d loaders=%d "
        "pos_clips=%d neg_clips=%d neg_zero_out=%s warnings=%d",
        roles.is_usable,
        roles.model_type,
        len(roles.sampler_ids),
        len(roles.model_loader_ids),
        len(roles.positive_clip_ids),
        len(roles.negative_clip_ids),
        roles.negative_is_zero_out,
        len(roles.warnings),
    )
    return roles


def extract_workflow_defaults(
    workflow: dict[str, Any],
    roles: WorkflowRoles,
) -> dict[str, Any]:
    """Extract the original default parameter values from an analyzed workflow.

    Reads the actual field values that the workflow author set on sampler,
    latent-image, and model-loader nodes.  These values represent the
    "designed-for" configuration of the template and can be surfaced in the UI
    so users know when they are deviating from the intended settings.

    Parameters
    ----------
    workflow:
        The same flat ``{node_id: {class_type, inputs}}`` dict that was passed
        to :func:`analyze_workflow`.
    roles:
        The :class:`WorkflowRoles` result from :func:`analyze_workflow`.

    Returns
    -------
    dict
        Keys: ``steps``, ``cfg``, ``sampler_name``, ``scheduler``,
        ``width``, ``height``, ``batch_size``, ``model_name``.
        Any value that cannot be determined is ``None``.
    """
    defaults: dict[str, Any] = {
        "steps": None,
        "cfg": None,
        "sampler_name": None,
        "scheduler": None,
        "width": None,
        "height": None,
        "batch_size": None,
        "model_name": None,
    }

    if not roles.is_usable or roles.primary_sampler_id is None:
        return defaults

    sampler_node = workflow.get(roles.primary_sampler_id)
    if not isinstance(sampler_node, dict):
        return defaults

    sampler_ct = sampler_node.get("class_type", "")
    sampler_inputs = sampler_node.get("inputs", {})

    # ── Steps, CFG, sampler_name, scheduler ─────────────────────────────────
    if sampler_ct in ("KSampler", "KSamplerAdvanced"):
        defaults["steps"] = sampler_inputs.get("steps")
        defaults["cfg"] = sampler_inputs.get("cfg")
        defaults["sampler_name"] = sampler_inputs.get("sampler_name")
        defaults["scheduler"] = sampler_inputs.get("scheduler")

    elif sampler_ct == "SamplerCustomAdvanced":
        # Steps + scheduler live on the sigmas/scheduler node upstream
        sigmas_ref = sampler_inputs.get("sigmas")
        if isinstance(sigmas_ref, list) and sigmas_ref:
            sigmas_id = str(sigmas_ref[0])
            sigmas_node = workflow.get(sigmas_id)
            if isinstance(sigmas_node, dict):
                sig_inputs = sigmas_node.get("inputs", {})
                defaults["steps"] = sig_inputs.get("steps")
                if sigmas_node.get("class_type") == "BasicScheduler":
                    defaults["scheduler"] = sig_inputs.get("scheduler")

        # Sampler name lives on the KSamplerSelect node upstream
        sampler_sel_ref = sampler_inputs.get("sampler")
        if isinstance(sampler_sel_ref, list) and sampler_sel_ref:
            sel_id = str(sampler_sel_ref[0])
            sel_node = workflow.get(sel_id)
            if isinstance(sel_node, dict) and sel_node.get("class_type") == "KSamplerSelect":
                defaults["sampler_name"] = sel_node.get("inputs", {}).get("sampler_name")

        # CFG lives on the guider node (only CFGGuider, not BasicGuider)
        guider_ref = sampler_inputs.get("guider")
        if isinstance(guider_ref, list) and guider_ref:
            guider_id = str(guider_ref[0])
            guider_node = workflow.get(guider_id)
            if isinstance(guider_node, dict) and guider_node.get("class_type") == "CFGGuider":
                defaults["cfg"] = guider_node.get("inputs", {}).get("cfg")

    # ── Width, height, batch_size from latent image node ────────────────────
    if roles.primary_latent_id:
        latent_node = workflow.get(roles.primary_latent_id)
        if isinstance(latent_node, dict):
            lat_inputs = latent_node.get("inputs", {})
            defaults["width"] = lat_inputs.get("width")
            defaults["height"] = lat_inputs.get("height")
            defaults["batch_size"] = lat_inputs.get("batch_size")

    # ── Model name from primary model loader ─────────────────────────────────
    if roles.primary_model_loader_id:
        loader_node = workflow.get(roles.primary_model_loader_id)
        if isinstance(loader_node, dict):
            loader_inputs = loader_node.get("inputs", {})
            defaults["model_name"] = (
                loader_inputs.get("ckpt_name")
                or loader_inputs.get("unet_name")
            )

    return defaults
