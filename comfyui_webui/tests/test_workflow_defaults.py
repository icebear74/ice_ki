"""Tests for workflow_analyzer.extract_workflow_defaults()."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Ensure the comfyui_webui package is on the path when running from tests/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import workflow_analyzer as _analyzer


# ---------------------------------------------------------------------------
# Shared workflow fixtures
# ---------------------------------------------------------------------------

def _ksampler_workflow() -> dict:
    """Standard KSampler workflow (checkpoint-based)."""
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "model.safetensors"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "pos", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "neg", "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 768, "batch_size": 2}},
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 42, "steps": 20, "cfg": 8.0,
                "sampler_name": "dpmpp_2m", "scheduler": "karras",
                "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
                "latent_image": ["4", 0],
            },
        },
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "out"}},
    }


def _unet_ksampler_workflow() -> dict:
    """UNETLoader + KSampler workflow with non-standard values."""
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux.safetensors"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "pos", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "neg", "clip": ["1", 1]}},
        "4": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 896, "height": 1152, "batch_size": 1}},
        "5": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": 3}},
        "6": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 77, "steps": 8, "cfg": 1.0,
                "sampler_name": "res_multistep", "scheduler": "simple",
                "model": ["5", 0], "positive": ["2", 0], "negative": ["3", 0],
                "latent_image": ["4", 0],
            },
        },
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["8", 0]}},
        "8": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": "out"}},
    }


def _custom_advanced_workflow() -> dict:
    """SamplerCustomAdvanced (FLUX-style) workflow."""
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux-dev.safetensors"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "pos", "clip": ["1", 1]}},
        "3": {"class_type": "RandomNoise", "inputs": {"noise_seed": 100}},
        "4": {
            "class_type": "BasicScheduler",
            "inputs": {"scheduler": "simple", "steps": 25, "model": ["1", 0], "denoise": 1.0},
        },
        "5": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "6": {
            "class_type": "CFGGuider",
            "inputs": {
                "model": ["1", 0], "positive": ["2", 0],
                "negative": ["2", 0], "cfg": 3.5,
            },
        },
        "7": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["3", 0], "guider": ["6", 0],
                "sampler": ["5", 0], "sigmas": ["4", 0],
                "latent_image": ["8", 0],
            },
        },
        "8": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["1", 2]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "out"}},
    }


def _custom_advanced_basic_guider_workflow() -> dict:
    """SamplerCustomAdvanced with BasicGuider (no cfg on guider)."""
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "model.safetensors"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "pos", "clip": ["1", 1]}},
        "3": {"class_type": "RandomNoise", "inputs": {"noise_seed": 200}},
        "4": {
            "class_type": "BasicScheduler",
            "inputs": {"scheduler": "beta", "steps": 30, "model": ["1", 0], "denoise": 1.0},
        },
        "5": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "dpmpp_2m"}},
        "6": {
            "class_type": "BasicGuider",
            "inputs": {"model": ["1", 0], "conditioning": ["2", 0]},
        },
        "7": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["3", 0], "guider": ["6", 0],
                "sampler": ["5", 0], "sigmas": ["4", 0],
                "latent_image": ["8", 0],
            },
        },
        "8": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 4}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["1", 2]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "out"}},
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class ExtractWorkflowDefaultsTests(unittest.TestCase):

    # ── KSampler (checkpoint) ────────────────────────────────────────────────

    def test_ksampler_checkpoint_all_fields(self) -> None:
        wf = _ksampler_workflow()
        roles = _analyzer.analyze_workflow(wf)
        defs = _analyzer.extract_workflow_defaults(wf, roles)

        self.assertEqual(defs["steps"], 20)
        self.assertAlmostEqual(defs["cfg"], 8.0)
        self.assertEqual(defs["sampler_name"], "dpmpp_2m")
        self.assertEqual(defs["scheduler"], "karras")
        self.assertEqual(defs["width"], 512)
        self.assertEqual(defs["height"], 768)
        self.assertEqual(defs["batch_size"], 2)
        self.assertEqual(defs["model_name"], "model.safetensors")

    # ── KSampler (UNet / AuraFlow-style) ─────────────────────────────────────

    def test_unet_ksampler_workflow(self) -> None:
        wf = _unet_ksampler_workflow()
        roles = _analyzer.analyze_workflow(wf)
        defs = _analyzer.extract_workflow_defaults(wf, roles)

        self.assertEqual(defs["steps"], 8)
        self.assertAlmostEqual(defs["cfg"], 1.0)
        self.assertEqual(defs["sampler_name"], "res_multistep")
        self.assertEqual(defs["scheduler"], "simple")
        self.assertEqual(defs["width"], 896)
        self.assertEqual(defs["height"], 1152)
        self.assertEqual(defs["batch_size"], 1)
        self.assertEqual(defs["model_name"], "flux.safetensors")

    # ── SamplerCustomAdvanced with CFGGuider ──────────────────────────────────

    def test_custom_advanced_cfg_guider(self) -> None:
        wf = _custom_advanced_workflow()
        roles = _analyzer.analyze_workflow(wf)
        defs = _analyzer.extract_workflow_defaults(wf, roles)

        self.assertEqual(defs["steps"], 25)
        self.assertAlmostEqual(defs["cfg"], 3.5)
        self.assertEqual(defs["sampler_name"], "euler")
        self.assertEqual(defs["scheduler"], "simple")
        self.assertEqual(defs["width"], 1024)
        self.assertEqual(defs["height"], 1024)
        self.assertEqual(defs["batch_size"], 1)
        self.assertEqual(defs["model_name"], "flux-dev.safetensors")

    # ── SamplerCustomAdvanced with BasicGuider (no cfg) ──────────────────────

    def test_custom_advanced_basic_guider_no_cfg(self) -> None:
        wf = _custom_advanced_basic_guider_workflow()
        roles = _analyzer.analyze_workflow(wf)
        defs = _analyzer.extract_workflow_defaults(wf, roles)

        self.assertEqual(defs["steps"], 30)
        self.assertIsNone(defs["cfg"])  # BasicGuider has no cfg
        self.assertEqual(defs["sampler_name"], "dpmpp_2m")
        self.assertEqual(defs["scheduler"], "beta")
        self.assertEqual(defs["width"], 512)
        self.assertEqual(defs["height"], 512)
        self.assertEqual(defs["batch_size"], 4)

    # ── Unusable workflow returns all-None defaults ───────────────────────────

    def test_unusable_workflow_returns_none_defaults(self) -> None:
        wf = {"1": {"class_type": "SaveImage", "inputs": {}}}
        roles = _analyzer.analyze_workflow(wf)
        self.assertFalse(roles.is_usable)
        defs = _analyzer.extract_workflow_defaults(wf, roles)
        for key in ("steps", "cfg", "sampler_name", "scheduler", "width", "height", "batch_size", "model_name"):
            self.assertIsNone(defs[key], f"{key} should be None for unusable workflow")

    # ── analyze_template_file returns defaults ────────────────────────────────

    def test_analyze_template_file_includes_defaults(self) -> None:
        import json
        import tempfile
        from pathlib import Path
        import template_registry as reg

        wf = _ksampler_workflow()
        with tempfile.TemporaryDirectory() as tmp:
            fpath = Path(tmp) / "test_workflow.json"
            fpath.write_text(json.dumps(wf), encoding="utf-8")
            result = reg.analyze_template_file(fpath)

        self.assertIn("defaults", result)
        defs = result["defaults"]
        self.assertEqual(defs["steps"], 20)
        self.assertEqual(defs["sampler_name"], "dpmpp_2m")
        self.assertEqual(defs["scheduler"], "karras")
        self.assertEqual(defs["width"], 512)
        self.assertEqual(defs["height"], 768)

    # ── _build_analysis_meta includes workflow_defaults ───────────────────────

    def test_build_analysis_meta_includes_workflow_defaults(self) -> None:
        import json
        import tempfile
        from pathlib import Path
        import template_registry as reg
        import main as webui_main

        wf = _unet_ksampler_workflow()
        with tempfile.TemporaryDirectory() as tmp:
            fpath = Path(tmp) / "unet_wf.json"
            fpath.write_text(json.dumps(wf), encoding="utf-8")
            validation = reg.analyze_template_file(fpath)

        meta = webui_main._build_analysis_meta(validation)
        self.assertIn("workflow_defaults", meta)
        wf_defaults = meta["workflow_defaults"]
        self.assertIsNotNone(wf_defaults)
        self.assertEqual(wf_defaults["steps"], 8)
        self.assertAlmostEqual(wf_defaults["cfg"], 1.0)
        self.assertEqual(wf_defaults["sampler_name"], "res_multistep")
        self.assertEqual(wf_defaults["scheduler"], "simple")


if __name__ == "__main__":
    unittest.main()
