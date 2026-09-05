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
│   ├── 01-storage.yaml              static hostPath PVs + PVCs
│   ├── 02-oobabooga.yaml            ConfigMap, Deployment, ClusterIP + NodePort
│   ├── 03-sillytavern.yaml          Deployment, ClusterIP + NodePort
│   ├── 04-character-extractor.yaml  Deployment + ClusterIP
│   ├── 05-comfyui.yaml              OPTIONAL, ships with replicas: 0
│   └── kustomization.yaml
├── extractor/                       Dockerfile, requirements, app code, schema, prompt
├── image-pipeline/                  ComfyUI workflow/prompt integration placeholders
└── scripts/                         build.sh, diagnose.sh, prepare-host-dirs.sh, verify-gpu.sh, verify-storage.sh, backup.sh
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
* A dedicated data filesystem mounted at `/mnt/aistack` (the current host uses
  a 1 TB volume there). Prepare it once and verify it before deploying:

  ```bash
  sudo textgenerator/scripts/prepare-host-dirs.sh
  textgenerator/scripts/verify-storage.sh
  ```

## 4. Storage

The stack uses **static hostPath PersistentVolumes** on the dedicated host
filesystem under `/mnt/aistack`. There is one PV per PVC. Each PV is pre-bound
to its claim with `claimRef`, and each PVC is pinned back to the intended PV
with `volumeName`. The `textgen-hostpath` StorageClass uses
`provisioner: kubernetes.io/no-provisioner`, `reclaimPolicy: Retain` and
`volumeBindingMode: Immediate`; it exists only so the claims never fall through
to a cluster default StorageClass.

| PVC | Size | Access | hostPath |
| --- | --- | --- | --- |
| `sillytavern-data-pvc` | 20Gi | RWO | `/mnt/aistack/sillytavern` |
| `oobabooga-models-pvc` | 500Gi | RWO | `/mnt/aistack/oobabooga/models` |
| `oobabooga-character-pvc` | 50Gi | RWO | `/mnt/aistack/oobabooga/character` |
| `shared-characters-pvc` | 20Gi | **RWX** | `/mnt/aistack/shared/characters` |
| `extractor-data-pvc` | 20Gi | RWO | `/mnt/aistack/extractor` |
| `comfyui-data-pvc` | 300Gi | RWO | `/mnt/aistack/comfyui` |

`reclaimPolicy: Retain` is set on the PVs. Deleting a PVC never deletes files
under `/mnt/aistack`. A retained PV moves to `Released` and will not bind again,
so `scripts/build.sh --clean --purge-data` deletes the PVCs **and** the PVs;
the on-disk files still remain.

> **Sizing.** The requested sizes are metadata only. hostPath does not enforce
> capacity or quotas; the real limit is the free space on `/mnt/aistack`. Check
> it with `df -h /mnt/aistack` before downloading large models.

> **Permissions.** Kubernetes never applies `fsGroup` to hostPath volumes, so
> host directory ownership decides whether the containers can write. Everything
> runs as uid/gid `1000`. `prepare-host-dirs.sh` creates the tree with
> `chown 1000:1000` and `chmod 0775`. `type: DirectoryOrCreate` means the
> kubelet creates anything missing as **root**, which is why the root
> `initContainer`s in `03-sillytavern.yaml` and `04-character-extractor.yaml`
> `chown` the mounts as a safety net. `subPath` parent directories must
> pre-exist for the same reason.

> `backups/`, `plugins/` and the third-party extensions directory are **not**
> under `dataRoot`: SillyTavern creates them relative to its working directory
> `/home/node/app`. They are mounted from the hostPath tree so they are writable
> and persistent – see §11 if you see `EACCES: permission denied, mkdir
> 'backups/'`.

If you move the storage root, edit **both** the `hostPath` values in
`k8s/01-storage.yaml` and `STORAGE_ROOT` in the scripts. They are not templated.

Local hostPath storage has no replication. It does not survive disk failure or a
move to another node – copy backups off the machine (§8).

Check the host directories and the resulting claims with:

```bash
textgenerator/scripts/verify-storage.sh
```

### Migrating from Longhorn

PVC fields such as `storageClassName` and `volumeName` are immutable. Existing
Longhorn-backed PVCs must be deleted before the hostPath manifests can bind:

1. Scale the deployments to zero.
2. Copy any data you want to keep out of the old volumes.
3. Delete the PVCs and the old PVs.
4. Run `sudo textgenerator/scripts/prepare-host-dirs.sh`.
5. Apply the stack again with `kubectl apply -k textgenerator/k8s`.

`scripts/build.sh --clean --purge-data` performs the delete step for the current
stack objects. If you forget this migration step, `kubectl apply` fails with an
immutable-field error, or the PVC stays `Pending` because it is still tied to the
old storage definition.

## 5. Deployment

One command does everything – prerequisite checks, building and importing the
extractor image, `kubectl apply -k`, waiting for the rollout and printing the
NodePort endpoints:

```bash
sudo textgenerator/scripts/build.sh
```

`sudo` is needed because the extractor image is imported into the K3s
containerd store.

| Option | Effect |
| --- | --- |
| `--clean` | Delete Deployments/Services/ConfigMaps before deploying. **PVCs are kept**, so no data is lost. |
| `--purge-data` | Only with `--clean`: also delete the PVCs and PVs. Files under `/mnt/aistack` stay in place; delete them manually with `sudo rm -rf /mnt/aistack/*` if you really want to erase data. Asks for confirmation. |
| `--clean-only` | Clean up and exit without deploying. |
| `--skip-build` | Do not rebuild the extractor image. |
| `--skip-verify` | Skip `verify-gpu.sh` / `verify-storage.sh`. |
| `--with-comfyui` | Scale the optional ComfyUI deployment to 1 (competes for the single GPU – see §7). |
| `-y`, `--yes` | Do not ask for confirmation on destructive actions. |

`build.sh` also runs `prepare-host-dirs.sh` automatically (via `sudo` when it is
not already running as root). `NAMESPACE`, `EXTRACTOR_TAG`, `ROLLOUT_TIMEOUT`
and `STORAGE_ROOT` can be overridden through the environment.

Typical redeploy after changing manifests or extractor code:

```bash
sudo textgenerator/scripts/build.sh --clean
```

Reset the Kubernetes storage objects while keeping the files on disk:

```bash
sudo textgenerator/scripts/build.sh --clean --purge-data
```

The individual steps are still available if you prefer to run them by hand:

```bash
sudo textgenerator/scripts/build-extractor-image.sh
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

`backup.sh` archives `/mnt/aistack` directly from disk, so no running pod is
needed:

```bash
sudo textgenerator/scripts/backup.sh
```

The destination defaults to `/var/backups/textgenerator`; pass another
directory as the first argument if needed. `KEEP` controls rotation and defaults
to `7` archives. By default the archive excludes large, reproducible data:
`oobabooga/models`, `comfyui/models` and `comfyui/output`. Set
`INCLUDE_MODELS=1` to include those directories anyway.

A live backup can catch a file mid-write. For a fully consistent snapshot, scale
the deployments down first:

```bash
kubectl -n ai-stack scale deploy --all --replicas=0
sudo textgenerator/scripts/backup.sh
kubectl -n ai-stack scale deploy --all --replicas=1
```

Copy the archive off this machine – hostPath storage has no redundancy. The
script prints the restore procedure for the archive it just wrote:

```bash
kubectl -n ai-stack scale deploy --all --replicas=0
sudo tar -xzf /var/backups/textgenerator/textgenerator-<timestamp>.tar.gz -C /mnt/aistack
sudo textgenerator/scripts/prepare-host-dirs.sh /mnt/aistack
kubectl -n ai-stack scale deploy --all --replicas=1
```

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
to edit `config.yaml` inside the volume by hand
(`kubectl -n ai-stack exec -it deploy/sillytavern -- vi config/config.yaml`).

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

### Pod stuck in `Pending` / `ContainerCreating`

Run the collector first – it prints pod events, PVC/PV state, StorageClass
presence, host directory ownership and node details in one go:

```bash
textgenerator/scripts/diagnose.sh
```

The decisive information is usually in the pod `Events:` section or in the PVC
/PV tables. With static hostPath storage, attach failures usually mean a pinned
PV is missing, a PVC does not match its PV, or the host directory is not writable
by uid/gid `1000`.

### PVC stays `Pending`

A PVC is pinned with `volumeName`, so the matching PV must already exist and its
size, accessModes and `storageClassName: textgen-hostpath` must match the claim.
Run:

```bash
textgenerator/scripts/diagnose.sh
kubectl -n ai-stack describe pvc <name>
```

### Immutable-field error / PVC stuck after the storage change

See §4, **Migrating from Longhorn**. Delete the old PVCs before applying the
hostPath manifests; `storageClassName` and `volumeName` cannot be changed in
place. `scripts/build.sh --clean --purge-data` deletes the current PVCs and PVs
while leaving `/mnt/aistack` untouched.

### `EACCES` / `Permission denied` in any pod

hostPath volumes ignore `fsGroup`, so ownership on the node decides. Run the
preparation script on the node that runs the pods and verify ownership:

```bash
sudo textgenerator/scripts/prepare-host-dirs.sh
stat -c '%u:%g' /mnt/aistack /mnt/aistack/shared/characters
```

Everything the containers write to must be owned by uid/gid `1000:1000`.

### PV stuck `Released` and not re-binding

This is the `Retain` policy. Delete the PV object and re-apply the manifests;
the files under `/mnt/aistack` stay in place:

```bash
kubectl delete pv <name>
kubectl apply -k textgenerator/k8s
```

### Node disk full

PVC sizes are not enforced for hostPath volumes. Check the real filesystem:

```bash
df -h /mnt/aistack
```

Free space by deleting or moving files under `/mnt/aistack`, or move the stack to
a larger mounted filesystem and update both `k8s/01-storage.yaml` and the
scripts' `STORAGE_ROOT`.

### `namespaces "ai-stack" not found`

Fixed: `build.sh` now creates the namespace before running the verification and
image-build steps. If you call the individual scripts by hand, apply
`k8s/00-namespace.yaml` first – `verify-gpu.sh` otherwise falls back to the
`default` namespace for its CUDA smoke test, and
`build-extractor-image.sh` skips the rollout restart when the deployment does
not exist yet.
