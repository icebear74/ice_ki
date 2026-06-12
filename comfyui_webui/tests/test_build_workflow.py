from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main as webui_main


def _checkpoint_workflow() -> dict[str, dict]:
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "base.safetensors"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "orig-pos", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "orig-neg", "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 42,
                "steps": 18,
                "cfg": 5.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
            },
        },
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "test"}},
    }


def _lumina_unet_workflow(*, use_zero_out_negative: bool = False) -> dict[str, dict]:
    negative_ref = ["4", 0]
    workflow = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "z_image_turbo_bf16.safetensors"}},
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "lumina2", "device": "default"},
        },
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "orig-pos", "clip": ["2", 0]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "orig-neg", "clip": ["2", 0]}},
        "6": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 896, "height": 1152, "batch_size": 1}},
        "7": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": 3}},
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 77,
                "steps": 8,
                "cfg": 1.0,
                "sampler_name": "res_multistep",
                "scheduler": "simple",
                "model": ["7", 0],
                "positive": ["3", 0],
                "negative": negative_ref,
                "latent_image": ["6", 0],
            },
        },
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["10", 0]}},
        "10": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        "11": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "test"}},
    }
    if use_zero_out_negative:
        workflow["5"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}}
        workflow["8"]["inputs"]["negative"] = ["5", 0]
    return workflow


class BuildWorkflowTests(unittest.TestCase):
    def test_default_template_applies_full_overrides(self) -> None:
        payload = webui_main.GenerateRequest(
            prompt_de="x",
            ollama_model="m",
            translated_prompt="unused",
            workflow_template="default",
            checkpoint="new_model.safetensors",
            steps=55,
            cfg=9.5,
            seed=123,
            width=640,
            height=768,
            sampler="ddim",
            scheduler="simple",
            image_count=2,
        )
        with patch.object(webui_main, "_load_default_workflow", return_value=copy.deepcopy(_checkpoint_workflow())):
            built, seed = webui_main._build_workflow(payload, "prompt+", "prompt-", req_id="t1")

        self.assertEqual(seed, 123)
        self.assertEqual(built["2"]["inputs"]["text"], "prompt+")
        self.assertEqual(built["3"]["inputs"]["text"], "prompt-")
        self.assertEqual(built["5"]["inputs"]["steps"], 55)
        self.assertEqual(built["5"]["inputs"]["cfg"], 9.5)
        self.assertEqual(built["5"]["inputs"]["sampler_name"], "ddim")
        self.assertEqual(built["5"]["inputs"]["scheduler"], "simple")
        self.assertEqual(built["4"]["inputs"]["width"], 640)
        self.assertEqual(built["4"]["inputs"]["height"], 768)
        self.assertEqual(built["4"]["inputs"]["batch_size"], 2)
        self.assertEqual(built["1"]["inputs"]["ckpt_name"], "new_model.safetensors")

    def test_imported_lumina_unet_preserves_pipeline_defaults(self) -> None:
        workflow = _lumina_unet_workflow()
        with tempfile.TemporaryDirectory() as tmp:
            template_dir = Path(tmp)
            (template_dir / "lumina.json").write_text(json.dumps(workflow), encoding="utf-8")
            payload = webui_main.GenerateRequest(
                prompt_de="x",
                ollama_model="m",
                translated_prompt="unused",
                workflow_template="lumina",
            )
            with (
                patch.object(webui_main._registry, "TEMPLATES_DIR", template_dir),
                patch.object(webui_main._registry, "get_template", return_value={"filename": "lumina.json"}),
                patch("main.os.urandom", return_value=b"\x00\x00\x00\x2a"),
            ):
                built, seed = webui_main._build_workflow(payload, "new-pos", "new-neg", req_id="t2")

        self.assertEqual(seed, 42)
        self.assertEqual(built["3"]["inputs"]["text"], "new-pos")
        self.assertEqual(built["4"]["inputs"]["text"], "new-neg")
        self.assertEqual(built["8"]["inputs"]["steps"], 8)
        self.assertEqual(built["8"]["inputs"]["cfg"], 1.0)
        self.assertEqual(built["8"]["inputs"]["sampler_name"], "res_multistep")
        self.assertEqual(built["8"]["inputs"]["scheduler"], "simple")
        self.assertEqual(built["6"]["inputs"]["width"], 896)
        self.assertEqual(built["6"]["inputs"]["height"], 1152)
        self.assertEqual(built["1"]["inputs"]["unet_name"], "z_image_turbo_bf16.safetensors")
        self.assertEqual(built["2"]["inputs"]["type"], "lumina2")
        self.assertEqual(built["8"]["inputs"]["seed"], 42)

    def test_imported_zero_out_negative_keeps_zero_out_chain(self) -> None:
        workflow = _lumina_unet_workflow(use_zero_out_negative=True)
        with tempfile.TemporaryDirectory() as tmp:
            template_dir = Path(tmp)
            (template_dir / "zero.json").write_text(json.dumps(workflow), encoding="utf-8")
            payload = webui_main.GenerateRequest(
                prompt_de="x",
                ollama_model="m",
                translated_prompt="unused",
                workflow_template="zero",
            )
            with (
                patch.object(webui_main._registry, "TEMPLATES_DIR", template_dir),
                patch.object(webui_main._registry, "get_template", return_value={"filename": "zero.json"}),
            ):
                built, _ = webui_main._build_workflow(payload, "new-pos", "new-neg", req_id="t3")

        self.assertEqual(built["3"]["inputs"]["text"], "new-pos")
        self.assertEqual(built["4"]["inputs"]["text"], "orig-neg")
        self.assertEqual(built["8"]["inputs"]["negative"], ["5", 0])

    def test_imported_unet_applies_only_explicit_overrides(self) -> None:
        workflow = _lumina_unet_workflow()
        with tempfile.TemporaryDirectory() as tmp:
            template_dir = Path(tmp)
            (template_dir / "explicit.json").write_text(json.dumps(workflow), encoding="utf-8")
            payload = webui_main.GenerateRequest(
                prompt_de="x",
                ollama_model="m",
                translated_prompt="unused",
                workflow_template="explicit",
                steps=12,
            )
            with (
                patch.object(webui_main._registry, "TEMPLATES_DIR", template_dir),
                patch.object(webui_main._registry, "get_template", return_value={"filename": "explicit.json"}),
                patch("main.os.urandom", return_value=b"\x00\x00\x00\x63"),
            ):
                built, seed = webui_main._build_workflow(payload, "new-pos", "new-neg", req_id="t4")

        self.assertEqual(seed, 99)
        self.assertEqual(built["8"]["inputs"]["steps"], 12)
        self.assertEqual(built["8"]["inputs"]["cfg"], 1.0)
        self.assertEqual(built["8"]["inputs"]["sampler_name"], "res_multistep")
        self.assertEqual(built["8"]["inputs"]["scheduler"], "simple")


if __name__ == "__main__":
    unittest.main()
