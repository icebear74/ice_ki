# image-pipeline (optional, ComfyUI)

Integration placeholders that connect the extracted **person profiles** to
image generation. Nothing in the text stack depends on this directory or on a
running ComfyUI instance.

## Relation to the existing `comfyui_webui/` in this repository

`comfyui_webui/` (repository root) is a FastAPI web UI that

* translates a German prompt to English via **Ollama**, and
* submits a ComfyUI workflow JSON to the ComfyUI API
  (`COMFYUI_BASE_URL`, default `http://127.0.0.1:8188`),
* using a workflow template plus a *mapping registry* that decides which
  workflow node receives the positive prompt, negative prompt, seed, etc.
  (see `comfyui_webui/workflow_template.json`, `mapping_registry.py`).

That application is therefore already a working ComfyUI **client**. When the
optional ComfyUI deployment (`../k8s/05-comfyui.yaml`) is enabled, it can be
pointed at the in-cluster service without any code change:

```bash
export COMFYUI_BASE_URL="http://comfyui.ai-stack.svc.cluster.local:8188"
# or, from outside the cluster, via the NodePort:
export COMFYUI_BASE_URL="http://<node-ip>:30188"
```

Containerising `comfyui_webui/` itself is intentionally **not** part of this
change - it is tracked as a follow-up so the text stack stays independent.

## Data flow

```
story text
   -> character-extractor (POST /extract)
        -> <shared>/characters/<Name>.json          (SillyTavern V2 card)
        -> <extractor>/profiles/<Name>.profile.json (person profile + visual_prompt)
             -> visual_prompt.positive / .negative
                  -> ComfyUI workflow (prompt node placeholders below)
                       -> <comfyui>/output/*.png
```

The extractor writes the image prompt into the profile file only. It never
calls ComfyUI itself, so image generation stays a manual/optional step.

## Prompt placeholders

`workflows/character_portrait.placeholder.json` is a **placeholder**, not a
runnable workflow: node ids and class types differ per checkpoint and per
ComfyUI version. Export a working workflow from the ComfyUI UI
("Save (API format)"), store it in the `workflows/` directory of
`comfyui-data-pvc` (mounted at `/opt/ComfyUI/workflows`), and replace these
tokens before submitting it:

| Token                  | Source field                          |
| ---------------------- | ------------------------------------- |
| `__POSITIVE_PROMPT__`  | `visual_prompt.positive` of the profile |
| `__NEGATIVE_PROMPT__`  | `visual_prompt.negative` of the profile |
| `__SEED__`             | any integer                            |
| `__CHECKPOINT__`       | file name under `comfyui/models/checkpoints` |

## Content policy

The image prompt wording is controlled by the single documented setting
`IMAGE_PROMPT_SAFETY_MODE` on the extractor deployment:

* `off` (default) - no extra wording is added; the uncensored behaviour of the
  chosen checkpoint is preserved.
* `sfw` - appends safe-for-work wording to the positive prompt and
  `nsfw, nudity` to the negative prompt.

Whatever a checkpoint or its own built-in filters do is a property of the
model, not of this repository.
