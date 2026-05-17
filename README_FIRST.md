# README_FIRST

## Worum es in diesem Projekt **wirklich** geht

Dieses Projekt ist **kein** generisches Face-Detection-, Speaker-Diarization- oder Videoanalyse-Projekt.

Das eigentliche Ziel ist ein **multimodales Persona-Extraktionssystem** für Serien / Filme.
Am Ende soll das System möglichst zuverlässig bestimmen können:

- **wer** etwas gesagt hat,
- **wie** diese Person spricht,
- **was** diese Person in der Serie weiß,
- welche **Charaktereigenschaften**, Gewohnheiten, Meinungen und Beziehungsmuster sie zeigt.

Diese Daten sollen später genutzt werden, um ein LLM mit der Persona der jeweiligen Figur / Person zu füttern, sodass sich das Modell konsistent wie diese Figur verhalten kann.

---

## Zielarchitektur

### Step 1 – Visuelle Personenerkennung
Über das Bild / Video soll erkannt werden:
- welche Person sichtbar ist,
- welche Gesichter zuverlässig derselben Figur / Person zugeordnet werden können,
- welche hochwertigen Gesichtsbeispiele als visuelle Referenz taugen.

**Wichtig:**
Step 1 dient **nicht** nur dazu, möglichst viele Gesichter zu finden.
Step 1 soll **hochwertige visuelle Identitätsanker** erzeugen, die später mit Audio und Text fusioniert werden können.

Optimierungsziel von Step 1:
- hohe Präzision,
- Identitätsreinheit,
- saubere Trennung der Figuren,
- hochwertige Face-Samples,
- möglichst wenig Cross-Contamination zwischen Personen.

Nicht das Hauptziel von Step 1:
- maximale Anzahl an Detections,
- hohe Recall-Zahlen um jeden Preis,
- viele schlechte Kandidatenbilder.

### Step 2 – Audio Fingerprinting / Sprechererkennung
Über die Stimme soll erkannt werden:
- wer spricht,
- welche Voiceprints / Sprecherprofile zu welcher Person gehören,
- wie sich Sprecher über Szenen und Episoden hinweg wiedererkennen lassen.

### Step 3 – Fusion von Bild + Audio
Bild und Audio werden zusammengeführt.

Beispiel:
- Wenn die Stimme stark nach Penny klingt
- und im Bild Penny zu sehen ist,
- dann ist die Zuordnung des gesprochenen Texts zu Penny mit sehr hoher Sicherheit möglich.

Diese Fusion ist der Kern, um später belastbar zu bestimmen:
- **wer was gesagt hat**
- auch über längere Episoden / Staffeln hinweg.

---

## Warum sauberes Facematching so wichtig ist

Facematching ist nur **ein Puzzleteil**, aber ein **kritisches**.
Wenn Step 1 unsauber ist und Identitäten vermischt, dann entstehen Folgefehler:

- falsche Person wird mit falscher Stimme gekoppelt,
- gesprochene Inhalte werden der falschen Figur zugeschrieben,
- Wissen und Charaktereigenschaften werden zwischen Figuren vermischt,
- das spätere Persona-LLM wird inkonsistent oder falsch.

Darum gilt:

**Weniger, aber hochreine und verlässliche Face-Samples sind wertvoller als viele noisy Detections.**

---

## Wie das System bewertet werden soll

Bitte dieses Projekt **nicht** primär nach folgenden Metriken bewerten:
- Anzahl der Face-Detections
- Anzahl der Tracks
- Anzahl der Kandidaten

Sondern nach:
- Wie sauber sind die Identitäten getrennt?
- Wie gut sind die visuellen Samples als Referenzmaterial?
- Wie gut lassen sich Bild und Audio später verknüpfen?
- Wie wenig manuelle Nacharbeit ist nötig?
- Wie sicher lässt sich am Ende bestimmen, wer was gesagt hat?

---

## Kurzfassung in einem Satz

Dieses Projekt soll aus Serien / Filmen multimodal extrahieren, **wer was gesagt hat**, um daraus Wissen, Sprachstil und Charakter einer Figur aufzubauen und später ein LLM konsistent auf diese Persona auszurichten.
