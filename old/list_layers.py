import torch

# Pfad zu deinem Checkpoint
ckpt_path = "/mnt/data/training/Universal/Mastermodell/Learn/checkpoints/vsr_checkpoint_emergency.pth"

try:
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    # Wir schauen in das model_state_dict
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    print("\n🔍 Gefundene Layer-Gewichte im Checkpoint:")
    print("-" * 50)
    
    count = 0
    for name in state_dict.keys():
        if "weight" in name:
            print(f"✅ {name}")
            count += 1
            
    print("-" * 50)
    print(f"Gesamt: {count} Gewichts-Layer gefunden.")
    
except Exception as e:
    print(f"❌ Fehler beim Laden: {e}")