# textgenerator – K3s stack for storytelling & roleplay

Deployment architecture for a **single-node Ubuntu 24.04 K3s host with one
NVIDIA Tesla P100 (16 GB)**:

| Component | Role | GPU | External access |
| --- | --- | --- | --- |
| **SillyTavern** | storytelling / roleplay front-end | no | NodePort `30800` |
| **Text Generation WebUI (oobabooga)** | model backend + HuggingFace downloader | **yes** (`nvidia.com/gpu: 1`) | Web UI on NodePort `30786`, API **ClusterIP only** |
| **character-extractor** | story text → SillyTavern V2 card + person profile | no | ClusterIP only (`port-forward`) |
| **ComfyUI** *(optional, `replicas: 0`)* | image generation | yes, on demand | NodePort `30188` |

Content scope: the stack is intended for **adult, fictional** storytelling and
roleplay. No content moderation, prompt filtering or censorship layer is
implemented. Model behaviour is a property of the model you load.

---

## 1. Architecture

```
                    NodePort 30800            NodePort 30786 (UI only)
                          │                            │
                    ┌─────▼──────┐   ClusterIP  ┌──────▼──────────────────┐
   browser ────────►│ SillyTavern│─────5000────►│ text-generation-webui   │──► GPU
                    └─────┬──────┘              │ 7860 UI  /  5000 API    │
                          │                     └──────▲──────────────────┘
              shared-characters-pvc                    │ ClusterIP 5000
                          │                            │
                    ┌─────▼───────────────┐            │
                    │ character-extractor │────────────┘
                    │  HTTP :8080         │
                    └─────┬───────────────┘
                          │ extractor-data-pvc (profiles + visual prompts)
                          ▼
                    ┌─────────────────────┐
                    │ ComfyUI (optional)  │──► GPU (serially, see §7)
                    └─────────────────────┘
```

Key decisions:

* **NodePort first.** No Ingress, no reverse proxy, no authentication yet –
  see §9 for the migration path. The manifests keep a plain `ClusterIP`
  Service next to every NodePort Service, so an Ingress can be added later
  without touching the Deployments.
* **The model API is never exposed via NodePort.** Only the SillyTavern UI and
  the diagnostic web interfaces (Oobabooga UI, ComfyUI UI) are.
* **Shared character storage.** SillyTavern and the extractor both mount
  `shared-characters-pvc`; neither pod reaches into the other's filesystem.
* **The extractor is a separate CPU service**, not a sidecar and not a
  ConfigMap script – reproducible dependencies, health checks, real error
  handling.

## 2. Files

```
textgenerator/
├── k8s/
│   ├── 00-namespace.yaml            namespace ai-stack
│   ├── 01-storage.yaml              StorageClass + hostPath PVs + PVCs
│   ├── 02-oobabooga.yaml            ConfigMap, Deployment, ClusterIP + NodePort
│   ├── 03-sillytavern.yaml          Deployment, ClusterIP + NodePort
│   ├── 04-character-extractor.yaml  Deployment + ClusterIP
│   ├── 05-comfyui.yaml              OPTIONAL, ships with replicas: 0
│   └── kustomization.yaml
├── extractor/                       Dockerfile, requirements, app code, schema, prompt
├── image-pipeline/                  ComfyUI workflow/prompt integration placeholders
└── scripts/                         prepare-host-dirs.sh, verify-gpu.sh, backup.sh
```

There is no `06-nodeports.yaml`: every Service lives next to its Deployment.

## 3. Prerequisites

* K3s on Ubuntu 24.04, NVIDIA driver + NVIDIA container toolkit installed, and
  a `RuntimeClass` named `nvidia`.
* Verify **before** deploying:

  ```bash
  sudo textgenerator/scripts/verify-gpu.sh
  ```

  It checks `nvidia-smi`, the `nvidia` RuntimeClass, the advertised
  `nvidia.com/gpu` capacity and runs an in-cluster CUDA smoke test.

## 4. Storage preparation

All persistent data lives under `/var/lib/k3s-ai-stack/` via **explicit
hostPath PersistentVolumes** (`storageClassName: textgen-hostpath`,
`persistentVolumeReclaimPolicy: Retain`), so every data set has a known,
backup-friendly location instead of an opaque `local-path` directory.

```bash
sudo textgenerator/scripts/prepare-host-dirs.sh          # /var/lib/k3s-ai-stack
```

| PVC | Host path | Mounted at |
| --- | --- | --- |
| `sillytavern-data-pvc` | `sillytavern/{config,data}` | `/home/node/app/config`, `/home/node/app/data` (`subPath`) |
| `oobabooga-models-pvc` | `oobabooga/models` | `/app/models` |
| `oobabooga-character-pvc` | `oobabooga/character/{characters,loras}` | `/app/characters`, `/app/loras` (`subPath`) |
| `shared-characters-pvc` | `shared/characters` | ST: `/home/node/app/data/default-user/characters`, extractor: `/data/characters` |
| `extractor-data-pvc` | `extractor/{profiles,raw}` | `/data/extractor` |
| `comfyui-data-pvc` | `comfyui/{models,input,output,user,workflows}` | `/opt/ComfyUI/...` (`subPath`) |

Local volumes survive pod restarts and image upgrades. They do **not** survive
a disk failure – see §8.

> Verify the container-internal paths against the image tags you actually
> deploy (SillyTavern moved its data directory in the past); adjust the
> `mountPath` values if a future tag differs.

## 5. Deployment

```bash
kubectl apply -k textgenerator/k8s
kubectl -n ai-stack get pods,svc,pvc
```

Then:

1. Open `http://<node-ip>:30786` (Oobabooga) → *Model* tab → download a model
   by HuggingFace repo id (e.g. a GGUF repo) → load it. First load can take
   several minutes; the startup probe allows up to 30 minutes.
2. Open `http://<node-ip>:30800` (SillyTavern) → *API Connections* → OpenAI
   compatible → `http://text-generation-webui.ai-stack.svc.cluster.local:5000/v1`.

### Building the extractor image

The extractor image is built locally and imported into the K3s image store:

```bash
docker build -t ice-ki/character-extractor:0.1.0 textgenerator/extractor
docker save ice-ki/character-extractor:0.1.0 | sudo k3s ctr images import -
kubectl -n ai-stack rollout restart deploy/character-extractor
```

## 6. Configuration knobs

| Setting | Where | Default | Meaning |
| --- | --- | --- | --- |
| `TEXTGEN_CONTEXT_SIZE` | ConfigMap `textgen-config` | `8192` | Maximum context length, passed to the backend as `--ctx-size` |
| `OOBABOOGA_API_BASE_URL` | ConfigMap `textgen-config` | internal Service DNS | Endpoint the extractor talks to |
| `CLI_ARGS` / `EXTRA_LAUNCH_ARGS` | `02-oobabooga.yaml` | `--api --listen ... --auto-devices --ctx-size $(TEXTGEN_CONTEXT_SIZE)` | Backend launch arguments |
| `EXTRACTOR_MODEL` | `04-character-extractor.yaml` | `""` (currently loaded model) | Model name sent to the API |
| `EXTRACTOR_MAX_TOKENS` | `04-character-extractor.yaml` | `2048` | Answer length limit of one extraction |
| `EXTRACTOR_ALLOW_OVERWRITE` | `04-character-extractor.yaml` | `false` | Allow replacing existing cards/profiles |
| `IMAGE_PROMPT_SAFETY_MODE` | `04-character-extractor.yaml` | `off` | **Only** content-policy switch: `off` (uncensored, default) or `sfw` |

### About “no context limit”

An unlimited context cannot be guaranteed and is not claimed here. The
effective limit is `min(context window of the loaded model, what fits in
16 GB VRAM)`. `TEXTGEN_CONTEXT_SIZE` exposes the configured **maximum** and is
the only place to change it; the extractor performs no arbitrary
application-side truncation of the input text – oversized inputs are rejected
by the backend rather than silently cut.

The context flag differs between text-generation-webui releases and loaders
(`--ctx-size` on current releases, `--n_ctx` on older llama.cpp loaders). Run
`--help` in the image you pinned and adjust the value in `02-oobabooga.yaml`
instead of using a flag the image does not accept.

## 7. GPU, Tesla P100 / CUDA caveats

* Oobabooga runs with `runtimeClassName: nvidia` and requests
  `nvidia.com/gpu: 1`.
* The extractor requests **no** GPU – it only uses the model through the API.
* The P100 is a Pascal card (compute capability 6.0). Recent CUDA/PyTorch
  builds may drop `sm_60`. Symptom: `no kernel image is available for
  execution on the device`. Fix: pin an older, CUDA 11.x based image tag in
  `02-oobabooga.yaml` (the tag there is an explicit pin, deliberately not
  `:latest`, so an upstream rebuild cannot break a working deployment).
* **ComfyUI and Oobabooga must normally run serially** on a single 16 GB card:

  ```bash
  kubectl -n ai-stack scale deploy/text-generation-webui --replicas=0
  kubectl -n ai-stack scale deploy/comfyui               --replicas=1
  ```

  GPU time-slicing in the NVIDIA device plugin would allow concurrent
  scheduling, but gives **no VRAM isolation** – an LLM plus an SDXL checkpoint
  will exhaust 16 GB. No second physical GPU is allocated anywhere.

## 8. Backups

```bash
sudo textgenerator/scripts/backup.sh /var/backups/k3s-ai-stack
```

Backs up SillyTavern config/chats, character cards, extracted profiles,
Oobabooga characters/LoRAs and ComfyUI workflows. Model files, checkpoints and
generated images are excluded (large and reproducible). Copy the archive to
another machine – a local PV does not protect against disk failure.

## 9. Migration path: Ingress + authentication

The NodePort services are additive. To move to authenticated access later:

1. Add an `Ingress` pointing at the existing ClusterIP Services
   (`sillytavern:8000`, optionally `text-generation-webui:7860`).
2. Add basic auth (Traefik middleware or an external reverse proxy) and TLS.
3. Remove the `*-nodeport` Services.
4. Optionally enable SillyTavern's own basic auth – the environment variables
   are prepared (commented) in `03-sillytavern.yaml` and reference a
   Kubernetes `Secret`. **No secrets are committed to this repository.**

Until then, expose the NodePorts to the trusted local network only.

## 10. Character extraction API

`character-extractor` is ClusterIP only:

```bash
kubectl -n ai-stack port-forward svc/character-extractor 8080:8080

curl -s http://127.0.0.1:8080/config
curl -s -X POST http://127.0.0.1:8080/extract \
  -H 'Content-Type: application/json' \
  -d '{"text": "<story text>", "character_name": "Elena"}'
```

| Endpoint | Purpose |
| --- | --- |
| `GET /healthz` | probe endpoint |
| `GET /config` | effective, secret-free configuration |
| `GET /profiles` | already extracted profiles |
| `POST /extract` | extract card + profile (`201`, `409` exists, `422` invalid input/answer, `502` backend down) |

It writes **two** artefacts:

1. `<shared>/characters/<Name>.json` – SillyTavern **V2** character card
   (`spec: chara_card_v2`, plus V1 top-level fields for older readers).
2. `<extractor>/profiles/<Name>.profile.json` – structured person profile:
   aliases, age, gender, species, occupation, appearance (height, build, skin,
   hair, eyes, distinguishing features), clothing, personality, speech style,
   background, relationships, tags – plus a generated `visual_prompt` and
   `metadata` (source, model, timestamp, confidence).

Rules enforced by the service:

* Missing information stays `null` / `[]` – **nothing is invented**; the
  extraction prompt contains no moderation instructions.
* Model answers are parsed defensively (code fences, surrounding prose),
  validated against the profile schema
  (`extractor/schemas/person_profile.schema.json`) and never written to disk
  unvalidated.
* File names are sanitised (no path traversal, ASCII, length limited), JSON is
  written deterministically (sorted keys, UTF-8) and **existing files are never
  overwritten** unless both `EXTRACTOR_ALLOW_OVERWRITE=true` and
  `"overwrite": true` are set.
* The generated image prompt is built **only** from extracted facts.

### Tests

```bash
cd textgenerator/extractor && python -m unittest discover -s tests
```

The core logic (parsing, normalisation, card/profile/prompt building, file
writing) is dependency-free and covered by unit tests; no GPU, model backend
or cluster is required to run them.
