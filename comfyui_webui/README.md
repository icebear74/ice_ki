# comfyui_webui

Lokale Python-Weboberfläche für:

1. Deutscher Prompt → Übersetzung nach Englisch über **Ollama**
2. Übergabe des übersetzten Prompts an **ComfyUI** zur Bildgenerierung
3. Benutzerverwaltung mit Rollen (admin / user)
4. Template-Freigabe-System: Admins können Workflow-Templates testen und freigeben

## K3s / Kubernetes mit Docker und Longhorn

Der Containerbetrieb benötigt **keine Python-/venv-Installation auf den Nodes**.
WebUI, ComfyUI und Ollama laufen in getrennten Containern. Die Dockerfiles bauen
die WebUI und ComfyUI (v0.38.0, festgelegter Commit, PyTorch 2.8/CUDA 12.6);
das offizielle Ollama-Image wird ebenfalls in die lokale Registry gespiegelt.
Benötigt werden Docker auf dem Build-Rechner, kubectl, ein laufender Cluster und
Longhorn. Für ComfyUI müssen NVIDIA-Treiber mit CUDA-12.6-Unterstützung,
Container Toolkit, NVIDIA Device Plugin und die RuntimeClass `nvidia` vorhanden
sein. Ressourcen und PVC-Größen im Manifest an die vorhandene Hardware anpassen.

**Tesla P100 (Pascal, `sm_60`):** Die CUDA-12.8-Wheels von PyTorch 2.8
unterstützen diese GPU nicht. Das Image verwendet deshalb explizit
`torch==2.8.0+cu126` und `torchvision==0.23.0+cu126`, auch während der
ComfyUI-Abhängigkeitsinstallation. Der Build prüft CUDA-Version und die
einkompilierte `sm_60`-Architektur ohne eine GPU auf dem Build-Rechner zu benötigen.
Zusätzliche Custom Nodes müssen ebenfalls diese Constraints beachten.
P100 unterstützt kein natives BF16; bei einem Workflow mit BF16-Anforderung
ein passendes FP16/FP32-Modell verwenden, ggf. ComfyUI mit `--force-fp32`
starten (höherer Speicherbedarf).

### Images bauen und deployen

Wie im Build-Skript unter `3dmodell` wird ohne Anmeldung nach
**`127.0.0.1:5000`** gepusht. Als erstes Argument wird die Registry-Adresse
angegeben, unter der **alle Cluster-Nodes** die Images abrufen können:

```bash
cd comfyui_webui
IMAGE_TAG=1 ./build-and-push.sh --deploy 192.168.1.10:5000 /tmp/deploy_comfyui.yaml
kubectl -n comfyui rollout status deployment/webui
kubectl -n comfyui rollout status deployment/comfyui
kubectl -n comfyui rollout status deployment/ollama
```

Bei einem bestehenden CUDA-12.8-Image ist ein **Rebuild mit neuem Tag** nötig,
z. B. `IMAGE_TAG=p100-cu126-2 ./build-and-push.sh 192.168.1.10:5000 /tmp/deploy_comfyui.yaml`,
danach das generierte Manifest erneut anwenden. Nur ein Pod-Neustart repariert
ein unverändertes Image nicht. Nach dem Rollout auf der GPU prüfen:

```bash
kubectl -n comfyui exec deployment/comfyui -- python -c \
  "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list()); print(torch.ones(1, device='cuda') + 1)"
```

Die LAN-Adresse ersetzen! `127.0.0.1:5000` als Pull-Adresse funktioniert nur,
wenn die Registry auf **jedem** Node lokal erreichbar ist. `k8s/deploy.yaml` ist
die Vorlage mit absichtlich ungültigen Image-Adressen; das Build-Skript rendert
alle drei Image-Adressen in die Ausgabedatei. Bei Updates einen neuen `IMAGE_TAG`
verwenden. `OLLAMA_IMAGE` kann das zu spiegelnde Image überschreiben
(Standard: `ollama/ollama:0.35.0`). Der Build lädt Quellen und Python-Pakete aus
dem Internet; die Cluster-Nodes ziehen anschließend ausschließlich aus der
lokalen Registry. Es wird kein Registry-Secret benötigt.

Mit `--deploy` wird nach dem Build das Manifest angewendet und der
Modelltransfer-API-Token automatisch als Kubernetes-Secret eingerichtet.
Vorhandene Tokens werden **beibehalten**, nicht bei jedem Build rotiert.
Ohne `--deploy` bleibt das Skript ein reiner Build-/Push-/Render-Schritt
ohne Clusterzugriff; anschließend `./deploy.sh /tmp/deploy_comfyui.yaml`
ausführen. Beide Deployment-Wege benötigen einen gültigen kubectl-Kontext
und Rechte zum Lesen/Anlegen von Secrets und Anwenden der Ressourcen.

Für eine HTTP-Registry auf **jedem K3s-Node** in
`/etc/rancher/k3s/registries.yaml` konfigurieren:

```yaml
mirrors:
  "192.168.1.10:5000":
    endpoint:
      - "http://192.168.1.10:5000"
```

Danach K3s bzw. `k3s-agent` neu starten. Bei anderen Kubernetes-Distributionen
den entsprechenden Container-Runtime-Mirror konfigurieren. Für Docker muss
eine nicht lokale HTTP-Registry ggf. als `insecure-registries` freigegeben sein.
Eine Registry ohne Authentifizierung/HTTPS ausschließlich in einem
vertrauenswürdigen, per Firewall geschützten Netz betreiben.

### Ports und erster Start

| Anwendung | Zugriff |
|---|---|
| WebUI | `http://<node-ip>:30080` (NodePort) |
| ComfyUI | `http://<node-ip>:30188` (NodePort) |
| Ollama | nur intern: `http://ollama:11434` (ClusterIP) |

### Fehlende Services oder belegte NodePorts

Das Build-Skript baut/pusht Images und erzeugt ein Manifest; **ohne `--deploy`**
führt es kein `kubectl apply` aus. Das generierte Manifest enthält alle drei Services,
einschließlich `comfyui` auf Port 8188 / NodePort 30188. Auch ein
abgestürzter GPU-Pod entfernt seinen Service nicht: Fehlt der Service selbst,
ist dies ein separates Deployment-/Apply-Problem, kein CUDA-Fehler.

```bash
./deploy.sh /tmp/deploy_comfyui.yaml
kubectl -n comfyui get service webui comfyui ollama
kubectl -n comfyui get pods -l app=comfyui
kubectl -n comfyui get endpointslices -l kubernetes.io/service-name=comfyui
```

**Die komplette Ausgabe von `kubectl apply` beachten.** Einzelne Ressourcen
können angelegt werden, obwohl andere fehlschlagen. Bei
`provided port is already allocated` sind die NodePorts clusterweit belegt:

```bash
kubectl get services -A -o wide
WEBUI_NODEPORT=31080 COMFYUI_NODEPORT=31188 IMAGE_TAG=p100-cu126-3 \
  ./build-and-push.sh 192.168.1.10:5000 /tmp/deploy_comfyui.yaml
./deploy.sh /tmp/deploy_comfyui.yaml
```

Freie Ports im Standardbereich 30000–32767 auswählen. Die Variablen ändern
nur externe NodePorts; interne Service-Namen und Ports bleiben gleich.
Ein Service ohne bereite Endpoints weist dagegen auf einen nicht bereiten Pod
hin (`kubectl -n comfyui describe pod -l app=comfyui` und Pod-Logs prüfen).
Für die genaue Ursache eines fehlenden Services werden die Apply-Fehler benötigt;
ein belegter NodePort ist ohne diese Ausgabe nur eine mögliche Ursache.

Die WebUI verwendet Service-DNS (`http://comfyui:8188`, `http://ollama:11434`),
nicht localhost. Ein Übersetzungsmodell wird bewusst **nicht automatisch**
heruntergeladen; z. B.:

```bash
kubectl -n comfyui exec deployment/ollama -- ollama list
./show-initial-password.sh
```

`show-initial-password.sh` zeigt Benutzername und Initialpasswort der **WebUI**
aus `/data/bootstrap_credentials.txt` an, ohne sie zu ändern. Nicht in öffentliche
Logs umleiten. Das ist nicht der Modelltransfer-API-Token aus `show-api-token.sh`;
ComfyUI selbst besitzt hier kein separates Login-Passwort. Nach einer
Passwortänderung zeigt die Datei weiterhin nur die ursprünglichen Zugangsdaten.

Das heruntergeladene Modell in der WebUI auswählen. Die Bootstrap-Datei nach
dem ersten Login löschen und das Passwort über die WebUI ändern:

```bash
kubectl -n comfyui exec deployment/webui -- rm /data/bootstrap_credentials.txt
```

Das Manifest unterdrückt das Bootstrap-Passwort im Pod-Log über
`COMFYUI_WEBUI_LOG_BOOTSTRAP_PASSWORD=false`; es bleibt nur in der geschützten
Datei verfügbar. Ohne diese Einstellung bleibt das lokale Verhalten
(Ausgabe im Terminal) erhalten. Zugriff auf Logs und PVC beschränken. ComfyUI bietet hier
**keine eigene Authentifizierung**. Beide NodePorts nur für vertrauenswürdige
LAN-Clients freigeben; für externe Nutzung einen authentifizierenden
HTTPS-Reverse-Proxy vorschalten. Nicht ungeschützt ins Internet stellen.

### Persistenz und Longhorn-Replikation

| PVC | Größe | Inhalt (jeweils unter `/data`) |
|---|---|---|
| `webui-data` | 10 GiB | Benutzer, Templates, Mappings, Modell-Aliase, Galerie, Logs |
| `comfyui-data` | 100 GiB | Modelle, Input/Output, Benutzer-Workflows, temporäre Dateien, Caches, Custom Nodes |
| `ollama-data` | 30 GiB | Ollama-Modelle und Konfiguration |

Alle PVCs verwenden `ReadWriteMany` (RWX) und die eigene StorageClass
`comfyui-longhorn-single` mit **`numberOfReplicas: "1"`**. Dies ist die
Longhorn-Datenreplikation; ein PVC selbst besitzt kein `replicas`-Feld.
Longhorn muss RWX-Share-Manager/NFS unterstützen; die Cluster-Nodes benötigen
einen NFSv4-Client. Die Deployments laufen weiterhin mit einer Instanz und
`Recreate`, um parallele Writer während Updates zu vermeiden. RWX bedeutet
nicht, dass die JSON-Datenbanken oder Transferverwaltung für mehrere
Anwendungsinstanzen ausgelegt sind.
Die WebUI verwendet einen Uvicorn-Worker; Sessions liegen im Arbeitsspeicher,
nach einem Neustart ist ein erneuter Login erforderlich.

**Eine Longhorn-Replik ist keine Hochverfügbarkeit:** Beim Ausfall des
Storage-Nodes können die Daten vorübergehend oder dauerhaft verloren gehen.
Backups einrichten. `reclaimPolicy: Retain` bewahrt Volumes nach dem Löschen
der PVCs; endgültige Löschung ist eine bewusste Administratoraktion.
Ohne Longhorn die StorageClass im Manifest ersetzen und deren Definition
entfernen; eine Replikation 1 kann dann nicht zugesichert werden.
StorageClass-Parameter gelten für **neu angelegte** Volumes, nicht rückwirkend.

Bestehende lokale WebUI-Daten vor dem ersten Start ins WebUI-PVC übernehmen.
Die Container verwenden UID/GID 1000. Ein eingeschränkter Root-Init-Container
setzt den Besitzer der jeweiligen PVC-Wurzel auf UID/GID 1000; `fsGroup`
wird wegen Longhorn-RWX/NFS nicht verwendet. Bereits vorhandene Unterverzeichnisse
werden nicht rekursiv verändert; migrierte Dateien
müssen für diesen Benutzer les- und schreibbar sein. `data/` wird nicht ins
Image kopiert, um Benutzerdateien und Zugangsdaten nicht mitzubauen.
Für Docker außerhalb Kubernetes ist entsprechend ein beschreibbares Volume
an `/data` zu mounten.

ComfyUI verwendet `--base-directory /data`. Modelldateien in die üblichen
Unterverzeichnisse legen, etwa `/data/models/checkpoints`,
`/data/models/diffusion_models`, `/data/models/text_encoders` und
`/data/models/vae`. Beispiel (zuerst passende Verzeichnisse anlegen):

```bash
kubectl -n comfyui exec deployment/comfyui -- mkdir -p /data/models/checkpoints
COMFY_POD=$(kubectl -n comfyui get pod -l app=comfyui -o jsonpath='{.items[0].metadata.name}')
kubectl -n comfyui cp ./mein-modell.safetensors "$COMFY_POD:/data/models/checkpoints/mein-modell.safetensors"
```

### ComfyUI Manager und Modell-Downloads

Der **ComfyUI Manager ist im ComfyUI-Image installiert** und wird beim Start
aktiviert. Für ComfyUI v0.38.0 nutzen wir dessen offizielles
`manager_requirements.txt` (Manager 4.2.2) statt eines zusätzlichen Git-Clones
nach `custom_nodes`. Die Legacy-Manager-Oberfläche ist aktiviert, damit
**Manager → Model Manager** für Modell-Downloads verfügbar ist.

Nach einem Update mit neuem Tag bauen und das generierte Manifest anwenden:

```bash
cd comfyui_webui
IMAGE_TAG=p100-manager-1 ./build-and-push.sh 192.168.1.10:5000 /tmp/deploy_comfyui.yaml
./deploy.sh /tmp/deploy_comfyui.yaml
kubectl -n comfyui rollout status deployment/comfyui
kubectl -n comfyui logs deployment/comfyui | grep -i manager
```

Die Registry-Adresse ersetzen. Danach **ComfyUI** unter
`http://<node-ip>:30188` öffnen (bzw. dem konfigurierten NodePort),
Browser neu laden und im Manager „Model Manager“ auswählen. Der Manager
gehört zur ComfyUI-Oberfläche, nicht zum Admin-Tab der separaten WebUI.
Heruntergeladene Modelle liegen unter `/data/models/` auf `comfyui-data`;
Manager-Konfiguration und Cache unter `/data/user/__manager/` ebenfalls auf
dem PVC. Anschließend die Modellliste in der WebUI aktualisieren.
Die Pods benötigen Internetzugriff auf Modellquellen; nicht jeder Download
ist ohne Freischaltung oder Zugang beim jeweiligen Anbieter verfügbar.

Beim ersten Containerstart wird `/data/user/__manager/config.ini` aus der
Image-Vorlage angelegt. Vorhandene Einstellungen werden **nicht überschrieben**.
Die Vorlage verwendet `security_level = normal` und
`network_mode = personal_cloud`, damit gelistete Modell-Downloads über den
LAN-NodePort möglich sind. Beliebige Git-URL- und Pip-Installationen bleiben
mit `allow_git_url_install = False` und `allow_pip_install = False` gesperrt.
Keine Absenkung auf `weak` ist nötig; bevorzugt vertrauenswürdige
`.safetensors`-Modelle verwenden.

Bei einem bestehenden Manager-Config mit 403 beim Download die genannten
Werte in `/data/user/__manager/config.ini` prüfen und anschließend den
ComfyUI-Pod neu starten. `personal_cloud` ist **keine Authentifizierung**:
Den ComfyUI-NodePort nur Administratoren im vertrauenswürdigen Netz zugänglich
machen oder einen authentifizierenden Reverse-Proxy vorschalten. Die
WebUI-Anmeldung schützt die separate ComfyUI-/Manager-Oberfläche nicht.

Manager und ComfyUI selbst werden über **Image-Rebuilds** aktualisiert,
nicht per „Update all“ im laufenden Container. Für CPU-only- oder andere
manuelle `args`-Overrides die Flags `--enable-manager` und
`--enable-manager-legacy-ui` beibehalten. Kein zusätzliches
`ComfyUI-Manager`-Verzeichnis auf dem PVC klonen: Der integrierte Manager
deaktiviert solche Doppelinstallationen.

Keine Modelle sind im Image enthalten. Nur vertrauenswürdige Modelle und
Custom Nodes verwenden. Zusätzliche Python-Abhängigkeiten von Custom Nodes
in einem abgeleiteten Dockerfile installieren und neu bauen, **nicht** nur
im laufenden Pod: Eine Pod-Neuerstellung würde diese Installation verlieren.
Custom Nodes auf dem PVC sind ausführbarer Code und benötigen dieselbe Prüfung
wie Änderungen am Image.

### Direkte Downloads, manueller Modell-Upload und Neustart

Das Image enthält den fest gepinnten Fork
[FNGarvin/ComfyUI-AutoModelDownloader](https://github.com/FNGarvin/ComfyUI-AutoModelDownloader)
(`419fd24ba57d20334351ddfff28ab93c84163a67`).
Die Integration ersetzt dessen ungeschützte Download-Endpunkte durch eine
lokal geprüfte Implementierung und ergänzt die aktuelle Missing-Models-Seitenleiste.
Der Custom Node wird aus einem zusätzlichen **Image-Pfad** geladen, nicht
ins PVC kopiert: `--base-directory /data` verdeckt ihn somit nicht.
Ein vorhandener eigener Clone dieses Add-ons auf `/data/custom_nodes` muss
vor dem Start entfernt/deaktiviert werden, damit nicht dessen ungeschützte
Endpunkte zusätzlich geladen werden.

Die neuen Transfer- und Neustart-Endpunkte sind ohne Token **gesperrt**.
Dasselbe Kubernetes-Secret versorgt ComfyUI und die WebUI; der WebUI-Server
gibt den Token niemals an den Browser weiter. Automatisch beim Deployment
einrichten (oder `build-and-push.sh --deploy` verwenden):

```bash
./deploy.sh /tmp/deploy_comfyui.yaml
./show-api-token.sh
```

`deploy.sh` erzeugt bei fehlendem Secret einen zufälligen 256-Bit-Token mit
OpenSSL. Die kurzlebige Token-Datei ist nur für den Besitzer zugänglich und
wird auch bei Fehlern entfernt. Der Token wird weder ins Deployment-Manifest
geschrieben noch in der Deployment-Ausgabe angezeigt. Ein vorhandenes Secret
ohne gültigen Token führt zum Abbruch statt zu einer stillen Überschreibung;
auch Zugriffsfehler werden nicht als fehlendes Secret behandelt.

`show-api-token.sh` liest den aktuellen Token aus `comfyui-model-api` und
zeigt ihn lokal zum Einfügen in den Passwortdialog an – nicht in öffentliche
Logs umleiten oder weitergeben. Für direkten Zugriff in ComfyUI wird er nur
im Arbeitsspeicher der Seite gehalten. Keine manuelle Token-Datei erforderlich.
`deploy.sh` hinterlegt die Secret-Version in den Pod-Vorlagen von ComfyUI
und WebUI. Die erste Einrichtung oder eine geänderte Secret-Version löst
einen Rollout aus, damit laufende Pods die Umgebungsvariablen neu laden.
Das funktioniert auch nach einem fehlgeschlagenen Apply beim erneuten Aufruf.
Dies kann laufende Generierungen/Transfers unterbrechen; ein unveränderter
Token löst durch die Annotation keinen zusätzlichen Rollout aus.
Nach manueller Token-Rotation `deploy.sh` erneut ausführen oder beide
Deployments neu starten.

Nach Rebuild mit neuem `IMAGE_TAG` und Apply:

* **ComfyUI / Missing Models:** Zusätzlicher Button „Auf PVC laden“ pro Modell,
  „Alle auf PVC laden“ für die aktuelle Seitenleiste. Die ursprünglichen
  Browser-Download-Buttons bleiben unverändert. Der Workflow muss
  Download-Metadaten (Name, Verzeichnis, URL) enthalten.
* **„Modell-Downloads (PVC)“:** Bei fehlenden Metadaten direkten HTTPS-Link,
  Dateinamen und Zielverzeichnis eingeben, Transfer freischalten, Fortschritt
  verfolgen. Für gated Hugging-Face-Modelle das separate Secret wie unten
  einrichten. Bei anderen Anbietern mit Login lokal herunterladen und über
  die WebUI hochladen; keine Anbieter-Tokens in URLs eintragen.
* **WebUI / Admin / Modell-Upload:** Als Admin eine Modelldatei auswählen,
  Zielverzeichnis wählen und hochladen. Der WebUI-Server streamt die Datei an
  ComfyUI; die Datei landet **auf dem ComfyUI-PVC**, nicht im Browser und nicht
  auf dem separaten WebUI-PVC. U. a. Checkpoints, Diffusionsmodelle,
  Textencoder, VAE, LoRAs, ControlNet, CLIP-Vision, Embeddings und Upscaler werden
  angeboten; ausführbare Custom Nodes sind keine Upload-Ziele.
* **ComfyUI neu starten:** Geschützter Admin-Button in der WebUI. Er beendet
  ausschließlich den ComfyUI-Prozess; Kubernetes startet den Container neu,
  Dateien auf dem PVC bleiben erhalten. Laufende Transfers blockieren den
  Neustart. Modelllisten-Aktualisierung genügt oft ohne Neustart; ein Neustart
  bricht laufende Bildgenerierung ab. Kein Kubernetes-ServiceAccount-Token
  oder Cluster-Admin-Recht wird hierfür benötigt.

Transfers schreiben zunächst eine temporäre Datei, veröffentlichen nur
vollständige Dateien und überschreiben keine vorhandenen Modelle.
Standardlimit pro Datei ist 40 GiB (`COMFYUI_MODEL_MAX_BYTES` in **beiden**
Deployments konsistent setzen). Genügend freien PVC-Speicher vorhalten.
Die WebUI nimmt manuelle Uploads zuerst als temporäre Multipart-Dateien unter
`/tmp` entgegen und leitet sie anschließend in begrenzten Blöcken weiter.
Für große Uploads braucht daher auch der WebUI-Pod ausreichend temporären
Speicher; das endgültige Modell liegt ausschließlich auf dem ComfyUI-PVC.
Downloads erlauben ausschließlich HTTPS zu freigegebenen Modellhosts;
private/lokale Zieladressen und unsichere Redirects werden abgelehnt.
Zusätzliche vertrauenswürdige Hosts lassen sich in ComfyUI über
`COMFYUI_MODEL_ALLOWED_HOSTS` freigeben. Nur vertrauenswürdige Modell-Dateien
verwenden: Die erlaubte Endung allein garantiert keine sichere Datei,
insbesondere bei Pickle-basierten `.pt`/`.pth`/`.ckpt`-Dateien.
Transfers nach einem Containerabbruch gegebenenfalls erneut starten;
Fortschritt/Warteschlange werden nicht persistent wiederaufgenommen.

#### Gated Hugging-Face-Modelle

Zuerst auf der Modellseite mit dem eigenen Hugging-Face-Account die Lizenz
akzeptieren bzw. Zugang beantragen. Ein Token umgeht diese Freigabe nicht.
Danach einen **Read-Token** erstellen, dessen Berechtigungen das gewünschte
gated Repository einschließen. Keine Account-Passwörter verwenden.

Der AutoDownloader liest `HF_TOKEN` ausschließlich im ComfyUI-Backend aus
dem separaten optionalen Secret **`comfyui-huggingface`**, Schlüssel `token`.
Der lokale Modelltransfer-Admin-Token (`comfyui-model-api`) bleibt unverändert;
im Browserdialog weiterhin diesen Admin-Token eingeben, nicht den HF-Token.
WebUI und Browser erhalten den HF-Token nicht.

Sichere Einrichtung im lokalen **Bash-Terminal** mit passendem kubectl-Kontext
(Token-Eingabe unsichtbar, kein Token als Kommandozeilenargument):

```bash
(
  set +x
  set -euo pipefail
  umask 077
  secret_dir=$(mktemp -d)
  trap 'rm -rf -- "$secret_dir"; unset hf_token' EXIT
  IFS= read -r -s -p 'Hugging Face Read-Token: ' hf_token
  printf '\n'
  [[ -n "${hf_token//[[:space:]]/}" ]] || { echo 'Token fehlt.' >&2; exit 1; }
  printf '%s' "$hf_token" > "$secret_dir/token"
  unset hf_token
  kubectl -n comfyui create secret generic comfyui-huggingface \
    --from-file="token=$secret_dir/token" --dry-run=client -o yaml \
    | kubectl -n comfyui apply --server-side --field-manager=hf-token-setup -f -
)
kubectl -n comfyui rollout restart deployment/comfyui
```

Zuvor das neue ComfyUI-Image mit neuem `IMAGE_TAG` bauen/pushen und das
aktualisierte Deployment anwenden, damit Backend und Secret-Referenz vorhanden
sind. Keine Änderung am WebUI-Image erforderlich. Zur Token-Rotation denselben
Einrichtungsschritt wiederholen und ComfyUI neu starten. Ohne Secret bleiben
öffentliche Downloads möglich. Kubernetes-Secrets sind nicht automatisch
verschlüsselt: Cluster-/Secret-Zugriff beschränken; niemals Token-Dateien oder
Secret-YAML ins Repository, in Tickets oder öffentliche Logs kopieren.

Als Download-URL einen HTTPS-`resolve`-Link zur Datei verwenden, z. B.
`https://huggingface.co/ORGANISATION/MODELL/resolve/main/datei.safetensors`.
Der HF-Bearer-Token wird pro Request nur an die **exakten** Hosts
`huggingface.co` und `hf.co` gesendet. Bei jedem Redirect wird das Ziel erneut
geprüft; CDN-/Subdomain-/Civitai-/zusätzliche Hosts erhalten **keinen** HF-Token.
Signierte, erlaubte CDN-Weiterleitungen werden ohne diesen Header verfolgt.
Cookies werden nicht weitergereicht. Ein zusätzlicher Download-Host erweitert
nie die Token-Empfängerliste.

Bei einem harten Abbruch können `*.part`-Dateien zurückbleiben. Diese erst
nach Stoppen des ComfyUI-Pods und Prüfung gezielt entfernen; laufende
Transfers oder bereits vollständige Modelle nicht löschen.

**Wichtig für vorhandene RWO-PVCs:** `accessModes` eines gebundenen Claims lässt
sich nicht einfach auf RWX umstellen. Nicht bestehende Claims löschen!
Zuerst Backups/Snapshots erstellen, Deployments stoppen, neue RWX-Claims mit
neuen Namen anlegen, Daten kontrolliert übernehmen und die Claim-Namen im
Manifest anpassen. Erst nach Prüfung auf die neuen Claims umstellen.
Auch bei RWX lädt die WebUI per HTTP hoch; sie benötigt keinen eigenen
Mount des ComfyUI-Volumes. Longhorn-Replikation bleibt unabhängig davon **1**.

Diese Token-Prüfung schützt nur die neuen Transfer-/Neustart-Endpunkte,
nicht die komplette ComfyUI- oder Manager-API. ComfyUI weiterhin ausschließlich
im geschützten Admin-Netz oder hinter einem authentifizierenden
HTTPS-Reverse-Proxy betreiben. Bei HTTP gehen Zugangsdaten unverschlüsselt
über das Netz; HTTPS ist für nicht vollständig vertrauenswürdige Netze nötig.

### Templates und Betriebsprüfung

Workflow-JSONs in der Admin-UI hochladen oder ins WebUI-PVC unter
`/data/templates/` kopieren und „lokale Templates entdecken“ ausführen.
Für Generierung den Workflow im **ComfyUI-API-Format** exportieren.
ComfyUI-Benutzer-Workflows auf dem separaten ComfyUI-PVC sind nicht automatisch
WebUI-Templates. Unbrauchbare lokale Workflows werden nicht automatisch
freigegeben; reine entdeckte Metadaten ohne Workflow-Datei können nicht
freigegeben oder stillschweigend durch das Standard-Template ersetzt werden.

`COMFYUI_WEBUI_DATA_DIR` steuert alle beschreibbaren WebUI-Dateien
(Standard im lokalen Betrieb: `data/` neben `main.py`, im Container: `/data`).
`/healthz` prüft nur die WebUI selbst, damit ein Ausfall von Ollama/ComfyUI nicht
zu Neustart-Schleifen der WebUI führt. Alle Deployments besitzen Startup-,
Readiness- und Liveness-Probes.

```bash
kubectl -n comfyui get pods,pvc,svc
kubectl -n comfyui logs deployment/comfyui
kubectl -n comfyui describe pod -l app=comfyui
```

Ollama lädt beim ersten Start automatisch **`qwen2.5:1.5b`** (kleines
mehrsprachiges Modell, rund 1 GB Download) für Deutsch→Englisch auf seinen PVC.
Vorhandene Modelle werden nicht erneut heruntergeladen oder gelöscht.
Der Init-Container benötigt dafür Internetzugriff auf die Ollama-Modellregistry;
bei Downloadfehlern bleibt der Pod im Init-Zustand statt ohne Modell zu starten.
Fortschritt: `kubectl -n comfyui logs deployment/ollama -c bootstrap-translation-model`.

Ollama läuft standardmäßig auf der CPU, sodass nur ComfyUI eine GPU belegt.
Das CPU-Limit ist **28 logische CPUs**; die WebUI übergibt `num_thread: 28`
und `num_gpu: 0` an beide Ollama-Endpunkte. 14 physische Kerne mit SMT ergeben
28 Threads, nicht 28 physische Kerne. Das Limit reserviert keine Kerne und
erzwingt keine Volllast; für kleine Modelle können 14 Threads sogar schneller
sein. `OLLAMA_NUM_THREADS` in der WebUI lässt sich entsprechend reduzieren.
Nur eine Ollama-Anfrage wird gleichzeitig verarbeitet (`OLLAMA_NUM_PARALLEL=1`).

Die WebUI liest installierte Modelle über `/api/tags`; früher wurde **kein**
initiales Modell heruntergeladen (nur ein manueller Pull dokumentiert).
Im Mapping-Editor wird die Liste erneut geladen, und für neue Mappings
`qwen2.5:1.5b` vorausgewählt, sobald es verfügbar ist. Bei bestehenden Mappings
das Modell selbst auswählen und speichern. Die Voreinstellung
`OLLAMA_DEFAULT_MODEL` muss bei Änderung in WebUI und Ollama-Init-Container
übereinstimmen. Das Modell ist ein allgemeines kleines Sprachmodell, kein
dedizierter Übersetzer; Übersetzungsqualität ist promptabhängig.

### Nur Ollama neu deployen

Kein neues Ollama-Image nötig: Der vorhandene Registry-Mirror wird weiterverwendet.
Die **aktuelle** Vorlage enthält die Bootstrap-Konfiguration. Auf dem Server
die LAN-Registry und den bereits gepushten Image-Tag einsetzen:

```bash
sed 's#registry.example.invalid:5000/comfyui-ollama:1#192.168.1.10:5000/comfyui-ollama:1#g' \
  /home/icebear/ice_ki/comfyui_webui/k8s/deploy.yaml \
  | kubectl apply -l app=ollama -f -
kubectl -n comfyui rollout status deployment/ollama --timeout=20m
kubectl -n comfyui exec deployment/ollama -- ollama list
```

Der Selektor wendet ausschließlich das Ollama-Deployment an; PVCs, ComfyUI
und WebUI bleiben unverändert. Vorhandener PVC und Service werden vorausgesetzt.
Nur Neustart ohne Konfigurationsänderung:
`kubectl -n comfyui rollout restart deployment/ollama`.
Für die neuen WebUI-Funktionen (Thread-Optionen und Modellvorauswahl) muss
zusätzlich das WebUI-Image neu gebaut/gepusht und dessen Deployment aktualisiert
werden; der Ollama-only-Schritt ändert die laufende WebUI nicht.

Für Ollama-GPU-Betrieb eine weitere GPU und dieselbe NVIDIA-Runtime plus
`nvidia.com/gpu`-Ressourcen konfigurieren; eine GPU wird nicht automatisch
zwischen Pods geteilt. Für CPU-only-ComfyUI `runtimeClassName` und beide
`nvidia.com/gpu`-Einträge entfernen und die vollständigen Container-Argumente
auf `["--listen", "0.0.0.0", "--port", "8188", "--disable-auto-launch",
"--base-directory", "/data", "--enable-manager", "--enable-manager-legacy-ui",
"--cpu"]` setzen (deutlich langsamer).

## Voraussetzungen

- Lokales **Ollama** (Standard: `http://127.0.0.1:11434`)
- Lokales **ComfyUI** mit API (Standard: `http://127.0.0.1:8188`)
- Python 3.10+

## Setup (venv)

```bash
cd comfyui_webui
./setup_env.sh
source venv/bin/activate
```

## Anwendung starten

```bash
uvicorn main:app --host 127.0.0.1 --port 8080 --reload
```

Dann im Browser öffnen:

- `http://127.0.0.1:8080`

## Konfiguration per Umgebungsvariablen

```bash
export OLLAMA_BASE_URL="http://127.0.0.1:11434"
export COMFYUI_BASE_URL="http://127.0.0.1:8188"
```

---

## Benutzerverwaltung & Authentifizierung

### Erster Start (Bootstrap)

Wenn beim ersten Start noch **keine** `data/users.json` existiert, legt die App
automatisch einen Admin-Account an. Das generierte Passwort wird:

1. im Terminal ausgegeben (in deutlich sichtbarer Box)
2. in `data/bootstrap_credentials.txt` gespeichert

**Beispiel-Ausgabe beim ersten Start:**

```
============================================================
  FIRST START – admin account created
  username : admin
  password : abc123XYZ...
  See comfyui_webui/data/bootstrap_credentials.txt
  Delete that file after first login!
============================================================
```

> **Wichtig:** Lösche `data/bootstrap_credentials.txt` nach dem ersten
> Login. Lege anschließend einen eigenen Admin-Account an oder ändere das
> Passwort (manuell in `data/users.json` oder per Admin-UI im nächsten Release).

### Rollen

| Rolle   | Beschreibung |
|---------|-------------|
| `admin` | Vollzugriff: Benutzerverwaltung, Template-Freigabe, alle generieren |
| `user`  | Kann generieren und nur freigegebene Templates sehen |

### Passwort-Speicherung

Passwörter werden **niemals im Klartext gespeichert**.  
Es wird PBKDF2-HMAC-SHA256 mit 600.000 Iterationen und zufälligem Salt verwendet
(Python-Stdlib `hashlib` – keine extra Abhängigkeiten).

### Datei-Speicherort

```
comfyui_webui/
  data/
    users.json               ← Benutzer (wird beim ersten Start angelegt)
    bootstrap_credentials.txt ← Einmaliges Bootstrap-Passwort (löschen!)
    templates.json           ← Template-Registry
    templates/               ← Optional: JSON-Workflow-Dateien
```

Die Dateien in `data/` sind in `.gitignore` eingetragen – sie werden nicht
ins Repository commited.

### Admin-UI

Admins sehen nach dem Login einen zusätzlichen **⚙ Admin**-Tab mit:
- **Template-Verwaltung:** Templates entdecken, freigeben, deaktivieren, löschen
- **Benutzerverwaltung:** Neue Benutzer anlegen, Benutzer deaktivieren/aktivieren

---

## Template-Freigabe-System

### Konzept

1. Admins entdecken oder registrieren Workflow-Templates
2. Nach Test können Templates als **freigegeben** markiert werden
3. Nur freigegebene + aktive Templates sind für normale Benutzer sichtbar

### Template-Quellen

- **`local`**: Manuell in der Admin-UI eingetragen
- **`comfyui`**: Über „ComfyUI-Templates entdecken" von der ComfyUI-Instanz abgerufen
  (nutzt `/api/workflow_templates` falls von ComfyUI bereitgestellt)

### Template-JSON-Dateien

Workflow-Templates können als JSON-Dateien in `data/templates/` abgelegt werden.
Der `filename`-Eintrag im Template-Datensatz zeigt auf diese Datei (relativ zu
`data/templates/`).

> **Wichtig:** Die App benötigt **keine** feste Node-ID-Konvention mehr.  
> Du kannst jeden von ComfyUI exportierten Workflow direkt als JSON-Datei
> ablegen und hochladen – die Analyse erkennt die Rollen automatisch.

---

## Workflow-Analyse und Validierung

### Was wird analysiert?

Beim Hochladen oder Entdecken einer Template-Datei analysiert die App den
Workflow-Graphen automatisch (Modul `workflow_analyzer.py`) und erkennt:

| Rolle | Erkannte Knotentypen |
|-------|----------------------|
| **Sampler** | `KSampler`, `KSamplerAdvanced` |
| **Checkpoint-Loader** | `CheckpointLoaderSimple`, `CheckpointLoader` |
| **UNet-Loader** | `UNETLoader`, `DiffusionModelLoader` |
| **Positiver Prompt** | `CLIPTextEncode`-Knoten im positiven Conditioning-Pfad |
| **Negativer Prompt** | `CLIPTextEncode` oder `ConditioningZeroOut` im negativen Pfad |
| **Latent-Quelle** | `EmptyLatentImage`, `EmptySD3LatentImage`, u. a. |
| **Decoder** | `VAEDecode`, `VAEDecodeTiled` |
| **Output** | `SaveImage`, `PreviewImage` |
| **img2img-Pfade** | `VAEEncode`, `LoadImage` (für zukünftige Unterstützung) |

### Analyse-Ergebnis in der Admin-UI

In der Template-Tabelle zeigt die Spalte **Analyse**:

- **✓ OK** – Workflow ist vollständig verwendbar, keine Warnungen
- **⚠ N Warnung(en)** – verwendbar, aber z. B. mehrdeutiger Sampler, mehrere CLIP-Knoten
- **✗ Nicht verwendbar** – kein Sampler gefunden, Parse-Fehler, o. ä.

Der Tooltip beim Hover zeigt Details zu Warnungen, Fehlern, Sampler- und Loader-Anzahl.

Mit dem **⟳**-Button wird die Analyse für ein bestehendes Template neu ausgeführt
(`GET /api/admin/templates/{name}/analysis`).

### Unterstützte Workflow-Typen

| Workflow-Typ | Unterstützt | Hinweise |
|---|---|---|
| Standard SD 1.x/2.x/XL | ✓ | CheckpointLoaderSimple + KSampler |
| FLUX / UNet-basiert | ✓ | UNETLoader/DiffusionModelLoader erkannt |
| Dual-CLIP (FLUX) | ✓ | Beide CLIPTextEncode-Knoten werden befüllt |
| `ConditioningZeroOut` negativ | ✓ | Negativ-Prompt wird nicht überschrieben |
| Mehrstufige Sampler-Pipelines | ⚠ | Ausgabe-Sampler (→ VAEDecode) wird bevorzugt |
| img2img / Inpainting | ⚠ | Strukturen erkannt, aber Parameter noch nicht vollständig injizierbar |
| Komplexe Conditioning-Graphen | ⚠ | Warnung wenn kein CLIPTextEncode erreichbar |

### Ausführungsverhalten bei importierten Templates

- Importierte Workflow-JSONs werden jetzt **strukturtreu** ausgeführt:
  - Die Pipeline-Struktur (Loader, Wrapper wie `ModelSamplingAuraFlow`, CLIP-/VAE-Ketten) bleibt unverändert.
  - Prompt-Injektion erfolgt nur auf den vom Graph-Analyzer erkannten positiven/negativen Pfaden.
- Für importierte Templates werden Sampler-/Latent-Defaults aus dem Template standardmäßig beibehalten.
  WebUI-Werte werden nur dann in diese Felder geschrieben, wenn sie explizit vom WebUI-Standard abweichen
  (z. B. geänderte Steps statt Default 30).
- Der Seed wird weiterhin pro Lauf gesetzt (fester Seed oder zufällig bei `-1`).
- Bei nicht eindeutigem positiven Prompt-Ziel wird die Generierung mit einem klaren Fehler abgebrochen
  statt stillschweigend den falschen Knoten zu überschreiben.

### Workflow-Aware Defaults & Warnungen (Erweitert-Tab)

Die WebUI zeigt im **Erweitert-Tab** automatisch an, welche Werte direkt aus dem
importierten Workflow-Template stammen, und warnt, wenn die aktuellen Einstellungen
davon abweichen:

- **Grüner Hinweis** unter einem Feld (z. B. `✓ Template-Standard: euler`): Der aktuell
  eingestellte Wert stimmt mit dem Workflow-Original überein – gute Voraussetzungen für
  korrekte Ergebnisse.
- **Gelber Hinweis** unter einem Feld (z. B. `⚠ Template-Standard: euler (aktuell: dpmpp_2m)`):
  Der aktuelle Wert weicht vom Template-Standard ab. Dies kann bei spezialisierten Workflows
  (z. B. AuraFlow, FLUX Turbo) zu schlechten oder falschen Ergebnissen führen.
- **Abweichungs-Warnung** oben im Erweitert-Tab: Zusammenfassung aller abweichenden Felder.
- Grüne Infoleiste: Zeigt alle Template-Standard-Werte auf einen Blick.

Diese Anzeige ist für importierte Templates aktiv. Beim eingebauten Standard-Template
werden keine Template-Warnungen angezeigt.

Im Admin-Bereich (Mapping-Formular) werden die aus dem Workflow extrahierten Standards
ebenfalls angezeigt, wenn ein Template ausgewählt wird – als Hilfestellung beim Anlegen
von Mappings.

**Extrahierte Felder**: `steps`, `cfg`, `sampler_name`, `scheduler`, `width`, `height`,
`batch_size`, `model_name` – für KSampler-, KSamplerAdvanced- und SamplerCustomAdvanced-
Workflows (inklusive FLUX-Stil-Pipelines mit BasicScheduler/KSamplerSelect/CFGGuider).

### Bekannte Einschränkungen

- **img2img**: Der Analyse-Code erkennt `VAEEncode`- und `LoadImage`-Pfade und meldet
  sie als „möglicher img2img-Workflow". Die tatsächliche Bildübergabe ist noch
  **nicht implementiert** – die Architektur ist jedoch darauf vorbereitet.
- Sehr ungewöhnliche Node-Typen für Sampler oder Loader (custom nodes) werden
  unter Umständen nicht erkannt.
- Bei Workflows mit mehr als einem Sampler wählt die WebUI den Sampler,
  dessen Ausgabe direkt in einen `VAEDecode`-Knoten fließt. Ist das nicht
  eindeutig, wird der erste gefundene Sampler verwendet (mit Warnung).

### Logging

Jede Generierungsanfrage wird in `data/generation.log` protokolliert (rotierend,
max. 10 MB, 5 Backups):

```
2024-01-15 12:00:00.123 | INFO | REQUEST id=a1b2c3d4 template='default' checkpoint='v1-5.safetensors' ...
2024-01-15 12:00:00.456 | INFO | ANALYSIS id=a1b2c3d4 usable=True sampler='5' model_type=checkpoint ...
2024-01-15 12:00:01.789 | DEBUG | SET_POSITIVE id=a1b2c3d4 clip_node='2' text='a futuristic cityscape...'
2024-01-15 12:00:01.791 | INFO | QUEUED id=a1b2c3d4 prompt_id=abc-123-...
```

Protokollierte Ereignisse:
- `REQUEST` – eingehende Generierungsparameter
- `TRANSLATED` – übersetzte Prompts
- `TEMPLATE` – gewähltes Template
- `ANALYSIS` – Graph-Analyse-Ergebnis (Rollen, Warnungen)
- `OVERRIDE_POLICY` – Modus für Mutationen (`full` bei Default-Template, sonst `preserve_imported`)
- `SET_POSITIVE` / `SET_NEGATIVE` / `SET_SAMPLER` / `SET_LATENT` / `SET_MODEL` – Mutationen
- `QUEUED` – erfolgreiche Übergabe an ComfyUI
- `COMFYUI_REJECT` / `COMFYUI_UNREACHABLE` – ComfyUI-Fehler

---

## Vorbereitung für Image2Image

Die Architektur ist für spätere img2img-Unterstützung vorbereitet:

- `workflow_analyzer.py` erkennt bereits `VAEEncode`, `LoadImage` und
  `VAEEncodeTiled`-Knoten und setzt `is_potentially_img2img = True`
- Das Analyse-Ergebnis wird in den Template-Metadaten gespeichert
- Der API-Endpunkt `GET /api/admin/templates/{name}/analysis` liefert vollständige
  Graph-Metadaten, die für eine spätere img2img-Implementierung genutzt werden können
- Zukünftig: `GenerateRequest` um `init_image`-Feld erweitern, `_build_workflow()`
  erkennt `primary_latent_id` bereits als `VAEEncode`-Quelle und kann dort das
  Eingabebild übergeben

---

## HTTPS-Empfehlung

### Warum kein automatisches HTTPS in der App?

Für eine **lokale, selbst gehostete** Anwendung empfehlen wir **keinen
eingebauten HTTPS-Terminator** direkt in der FastAPI-App, da:

- Selbstsignierte Zertifikate im Browser Warnungen erzeugen
- Zertifikatserneuerung (Let's Encrypt) von außen erreichbar sein muss
- Ein Reverse Proxy flexibler und sicherer ist

### Empfohlener Ansatz: Nginx oder Caddy als Reverse Proxy

**Option A – Caddy (einfachstes HTTPS mit automatischem Zertifikat):**

```bash
# Caddyfile
yourdomain.local {
    reverse_proxy 127.0.0.1:8080
}
```

Caddy übernimmt automatisch HTTPS, falls die Domain öffentlich erreichbar ist.

**Option B – nginx mit selbstsigniertem Zertifikat für reines LAN:**

```bash
# Zertifikat erzeugen
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem \
  -days 365 -nodes -subj "/CN=localhost"

# nginx.conf (Auszug)
server {
    listen 443 ssl;
    ssl_certificate     /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

**Option C – uvicorn mit SSL direkt (für schnelle Tests):**

```bash
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem \
  -days 365 -nodes -subj "/CN=localhost"

uvicorn main:app --host 0.0.0.0 --port 8443 \
  --ssl-keyfile key.pem --ssl-certfile cert.pem
```

> Hinweis: Bei selbstsignierten Zertifikaten zeigt der Browser eine
> Sicherheitswarnung. Diese kannst du für lokale Entwicklung akzeptieren.
> Für Produktionsbetrieb im LAN empfehlen wir Option A oder B.

---

## Funktionen

- Eingabefeld für deutschen Prompt
- Eingabefeld für Negative Prompt
- Ollama-Modell auswählbar/eingebbar
- Verfügbare Ollama-Modelle abrufen
- Bildparameter einstellbar: Steps, CFG, Seed, Width, Height, Sampler, Scheduler, Anzahl Bilder
- ComfyUI-Checkpoint auswählbar/eingebbar
- **Workflow-Template auswählbar** (nur freigegebene Templates für normale Benutzer)
- Anzeige des übersetzten Prompts vor der Generierung
- Text in Anführungszeichen wie `"Hallo Welt"` bleibt bei der Übersetzung als exakter sichtbarer Schriftzug erhalten
- Anzeige der generierten Bilder in der UI
- **Login-/Logout-Funktion**
- **Admin-Panel:** Template-Freigabe + Benutzerverwaltung

## Hinweise zu ComfyUI-Integrationspunkten

Die App nutzt bewusst einfache, robuste Standard-Endpunkte:

- Prompt senden: `POST /prompt`
- History abrufen: `GET /history/{prompt_id}`
- Bild abrufen: `GET /view`
- Checkpoints bevorzugt über `GET /object_info/CheckpointLoaderSimple`
- Fallback für Checkpoints: `GET /models` (falls verfügbar)
- Template-Entdeckung: `GET /api/workflow_templates` (falls von ComfyUI bereitgestellt)

## Workflow-Template

Die App verwendet ein einfaches internes Standard-Workflow-Template mit diesen Knoten:

- `CheckpointLoaderSimple`
- `CLIPTextEncode` (positiv/negativ)
- `EmptyLatentImage`
- `KSampler`
- `VAEDecode`
- `SaveImage`

Wenn dein ComfyUI-Setup andere Knoten/Parameter benötigt, passe `workflow_template.json` an.  
Die App lädt dieses JSON automatisch (Fallback ist das interne Default-Template in `main.py`).

---

## Ollama auf eine bestimmte Grafikkarte begrenzen

Ollama wählt standardmäßig die erste verfügbare GPU. Mit der Umgebungsvariable
`CUDA_VISIBLE_DEVICES` kannst du den Prozess auf eine bestimmte Karte einschränken.

### GPU-Index herausfinden

```bash
# nvidia-smi zeigt alle GPUs mit Index 0, 1, 2, …
nvidia-smi -L
```

Beispielausgabe:
```
GPU 0: NVIDIA GeForce RTX 3090 (UUID: …)
GPU 1: NVIDIA Tesla P100 (UUID: …)
```

### Ollama auf GPU 1 begrenzen

```bash
CUDA_VISIBLE_DEVICES=1 ollama serve
```

Oder dauerhaft als systemd-Service-Override:

```bash
sudo systemctl edit ollama
```

Inhalt (anpassen):
```ini
[Service]
Environment="CUDA_VISIBLE_DEVICES=1"
```

Dann neu starten:

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

### Mehrere GPUs zulassen (z. B. 0 und 2)

```bash
CUDA_VISIBLE_DEVICES=0,2 ollama serve
```

### GPU vollständig deaktivieren (CPU-only)

```bash
CUDA_VISIBLE_DEVICES="" ollama serve
```

> **Hinweis:** Die Umgebungsvariable muss gesetzt sein, **bevor** Ollama startet.
> Wird sie nachträglich geändert, muss Ollama neugestartet werden.
