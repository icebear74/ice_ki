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
| `sillytavern-data-pvc` | `sillytavern/{config,data,backups,plugins,extensions}` | `/home/node/app/{config,data,backups,plugins,public/scripts/extensions/third-party}` (`subPath`) |
| `oobabooga-models-pvc` | `oobabooga/models` | `/app/models` |
| `oobabooga-character-pvc` | `oobabooga/character/{characters,loras}` | `/app/characters`, `/app/loras` (`subPath`) |
| `shared-characters-pvc` | `shared/characters` | ST: `/home/node/app/data/default-user/characters`, extractor: `/data/characters` |
| `extractor-data-pvc` | `extractor/{profiles,raw}` | `/data/extractor` |
| `comfyui-data-pvc` | `comfyui/{models,input,output,user,workflows}` | `/opt/ComfyUI/...` (`subPath`) |

Local volumes survive pod restarts and image upgrades. They do **not** survive
a disk failure – see §8.

> `backups/`, `plugins/` and the third-party extensions directory are **not**
> under `dataRoot`: SillyTavern creates them relative to its working directory
> `/home/node/app`. They are mounted so they are writable and persistent – see
> §11 if you see `EACCES: permission denied, mkdir 'backups/'`.

> Do not skip `prepare-host-dirs.sh`. It also pre-creates
> `sillytavern/data/default-user/characters`, the mount point of the shared
> character PVC. If that path does not exist, the kubelet creates the parent
> directories as `root` and SillyTavern (uid 1000) cannot write its user data.
> `fsGroup` does **not** fix this – Kubernetes does not apply it to `hostPath`
> volumes, so host ownership has to be correct.

> Verify the container-internal paths against the image tags you actually
> deploy (SillyTavern moved its data directory in the past); adjust the
> `mountPath` values if a future tag differs.

## 5. Deployment

Build the extractor image **first** – it exists in no registry, and the
deployment uses `imagePullPolicy: Never`:

```bash
sudo textgenerator/scripts/build-extractor-image.sh
```

Then deploy the stack:

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

`scripts/build-extractor-image.sh` builds the image and imports it into the
K3s containerd image store (it uses `nerdctl` when Docker is not installed).
Equivalent manual steps:

```bash
docker build -t ice-ki/character-extractor:0.1.0 textgenerator/extractor
docker save ice-ki/character-extractor:0.1.0 | sudo k3s ctr images import -
kubectl -n ai-stack rollout restart deploy/character-extractor
```

Repeat this after every change to `textgenerator/extractor/`.

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
| `PUID` / `PGID` | `03-sillytavern.yaml` | `1000` | uid/gid the SillyTavern server drops to; must match the host directory owner |
| `SILLYTAVERN_LISTEN` / `SILLYTAVERN_WHITELISTMODE` | `03-sillytavern.yaml` | `true` / `false` | Accept connections from the LAN (**requires image ≥ 1.13.0**, see §11) |
| `SILLYTAVERN_SECURITYOVERRIDE` | `03-sillytavern.yaml` | `true` | Allows listening on non-localhost without auth. Set to `false` once basic auth or an authenticating Ingress is in place |

Any `config.yaml` key can be set this way: upper-case the key path and replace
dots with underscores (`backups.common.numberOfBackups` →
`SILLYTAVERN_BACKUPS_COMMON_NUMBEROFBACKUPS`).

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
   Once auth is active, set `SILLYTAVERN_SECURITYOVERRIDE` back to `"false"`.

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

## 11. Troubleshooting

### `sillytavern` CrashLoopBackOff: `EACCES: permission denied, mkdir 'backups/'`

```
Error: EACCES: permission denied, mkdir 'backups/'
    at ensurePublicDirectoriesExist (/home/node/app/src/users.js)
```

Cause: the upstream image contains **no `USER` directive** – it starts as root,
and `/home/node/app` is owned `root:node` without group write. SillyTavern
creates `backups/` (a hardcoded path relative to its working directory, *not*
under `dataRoot`) on every start. A pod `securityContext` with
`runAsUser: 1000` therefore makes startup fail.

Fixed in this repository by:

* pinning image **1.18.0** and using its `PUID`/`PGID` mode (added in 1.17.0):
  the entrypoint starts as root, corrects the ownership of the mounted
  directories and then drops the server to uid/gid 1000 via `su-exec`. The pod
  log shows `Mode: PUID/PGID (UID:1000 GID:1000)` and only `tini` stays root;
* removing `runAsUser`/`runAsGroup` from the pod `securityContext` (setting
  them would force "Strict Non-Root" mode, in which the entrypoint cannot fix
  permissions);
* mounting `backups`, `plugins` and the third-party extensions directory so
  they are writable **and** survive restarts.

If you pin an older tag, be aware that `SILLYTAVERN_*` environment variables
only exist from **1.13.0** on. On 1.12.x they are silently ignored, the
default `whitelistMode: true` stays active and LAN clients get
"Forbidden: connection attempt from IP that is not whitelisted" – you then have
to edit `/var/lib/k3s-ai-stack/sillytavern/config/config.yaml` by hand.

### `sillytavern` exits immediately after "listening"

```
Your current SillyTavern configuration is insecure (listening to non-localhost).
Enable whitelisting, basic authentication or user accounts.
```

From 1.13 on SillyTavern **refuses to start** when it listens on a
non-localhost address while whitelist, basic auth and user accounts are all
disabled. `SILLYTAVERN_SECURITYOVERRIDE: "true"` in `03-sillytavern.yaml`
acknowledges this. Keep the NodePort on the trusted LAN and switch to basic
auth or an authenticating Ingress (§9) as soon as possible.

### `character-extractor` ErrImagePull / ErrImageNeverPull

`ice-ki/character-extractor:0.1.0` is built locally and exists in no registry.
Build and import it, then restart the deployment:

```bash
sudo textgenerator/scripts/build-extractor-image.sh
```

Verify the image is present in the containerd store K3s actually uses:

```bash
sudo k3s ctr images ls -q | grep character-extractor
```

`ErrImageNeverPull` means the image is still missing (the deployment uses
`imagePullPolicy: Never` so containerd never contacts Docker Hub).

### `text-generation-webui` stuck in `PodInitializing`

Normal on the first start: the image is several GB, and an init container from
a cluster-wide mutating webhook (for example `k8tz`) may run first. Watch the
progress with:

```bash
kubectl -n ai-stack describe pod -l app.kubernetes.io/name=text-generation-webui | tail -20
kubectl -n ai-stack logs -f deploy/text-generation-webui
```

If it later crashes with `no kernel image is available for execution on the
device`, the image lacks Pascal (`sm_60`) support – see §7.

### Checking permissions after the fact

```bash
sudo ls -ln /var/lib/k3s-ai-stack/sillytavern
```

Everything the containers write to must be owned by uid/gid `1000`. Re-run
`sudo textgenerator/scripts/prepare-host-dirs.sh` if directories are missing,
and `sudo chown -R 1000:1000 /var/lib/k3s-ai-stack/sillytavern` if the kubelet
created some as root.
