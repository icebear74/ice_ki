import torchvision.models as models
import torch
import os

print("\n--- VGG16 Download Test ---")
print(f"PyTorch Version: {torch.__version__}")
print("Versuche VGG16 (ImageNet Weights) zu laden...")

try:
    # Dieser Befehl löst den Download aus, wenn die Datei noch nicht im Cache ist
    vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    
    print("\n✅ ERFOLG: VGG16 wurde erfolgreich geladen!")
    
    # Pfad finden, wo es liegt
    cache_dir = os.path.expanduser('~/.cache/torch/hub/checkpoints/')
    print(f"📂 Speicherort Cache: {cache_dir}")
    
    if os.path.exists(cache_dir):
        files = os.listdir(cache_dir)
        vgg_files = [f for f in files if 'vgg16' in f]
        if vgg_files:
            print(f"   Gefundene Datei: {vgg_files[0]}")
            size_mb = os.path.getsize(os.path.join(cache_dir, vgg_files[0])) / (1024*1024)
            print(f"   Größe: {size_mb:.2f} MB")
        else:
            print("   (Datei ist geladen, aber Dateiname im Cache nicht eindeutig identifizierbar)")
    
except Exception as e:
    print(f"\n❌ FEHLER: Der Download hat nicht geklappt.")
    print(f"Grund: {e}")
    print("\nHinweis: Prüfe deine Internetverbindung oder Firewall.")

print("\n---------------------------")
