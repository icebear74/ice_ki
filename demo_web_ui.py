#!/usr/bin/env python3
"""
Web UI Demo - Demonstriert das vollständige Web-Monitoring-Interface
Simuliert Training mit ALLEN Metriken und Layer-Aktivitäten
"""

import sys
import os
import time
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vsr_plus_plus.systems.web_ui import WebMonitoringInterface


def demo_complete_web_ui():
    """Demonstriert Web UI mit vollständigen Mock-Daten"""
    print("\n" + "="*70)
    print("VSR++ Vollständiges Web UI Demo")
    print("="*70)
    
    # Start web interface
    web_ui = WebMonitoringInterface(port_num=5051, refresh_seconds=3)
    
    print(f"\n✓ Öffne deinen Browser:")
    print(f"   http://localhost:5051/monitoring")
    print(f"\n✓ Features:")
    print(f"   • ALLE Metriken aus Terminal-GUI")
    print(f"   • Layer-Aktivitäts-Balken")
    print(f"   • Quality-Metriken")
    print(f"   • Adaptive Weights")
    print(f"   • TensorBoard-Link")
    print(f"   • Validation-Button")
    print(f"   • Einstellbare Auto-Aktualisierung")
    print(f"\n✓ Drücke Ctrl+C zum Stoppen\n")
    
    # Simulate layer names
    layer_names = [
        'Enc Block 1', 'Enc Block 2', 'Enc Block 3',
        'Fusion 1', 'Fusion 2',
        'Dec Block 1', 'Dec Block 2', 'Dec Block 3',
        'Final Fusion'
    ]
    
    iteration = 0
    base_loss = 0.05
    
    try:
        while True:
            iteration += 1
            
            # Simulate decreasing loss
            total_loss = base_loss * (1.0 - min(iteration / 10000.0, 0.9))
            l1_loss = total_loss * 0.4
            ms_loss = total_loss * 0.35
            grad_loss = total_loss * 0.25
            
            # Simulate varying learning rate
            learn_rate = 0.0001 * max(0.1, 1.0 - iteration / 20000.0)
            
            # Simulate quality metrics
            lr_quality = min(0.75, 0.5 + iteration / 15000.0)
            ki_quality = min(0.92, 0.5 + iteration / 10000.0)
            improvement = ki_quality - lr_quality
            
            # Simulate layer activities
            layer_activities = {}
            for idx, layer_name in enumerate(layer_names):
                # Add some variation
                base_activity = 0.3 + (idx / len(layer_names)) * 0.6
                variation = random.uniform(-0.1, 0.1)
                layer_activities[layer_name] = min(0.99, max(0.01, base_activity + variation))
            
            # Calculate ETA
            remaining_steps = 100000 - iteration
            eta_seconds = remaining_steps * 0.42
            hours = int(eta_seconds // 3600)
            minutes = int((eta_seconds % 3600) // 60)
            seconds = int(eta_seconds % 60)
            eta_total = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
            epoch_remaining = 1000 - (iteration % 1000)
            epoch_eta_sec = epoch_remaining * 0.42
            epoch_mins = int(epoch_eta_sec // 60)
            epoch_secs = int(epoch_eta_sec % 60)
            epoch_eta = f"00:{epoch_mins:02d}:{epoch_secs:02d}"
            
            # Update web UI with COMPLETE data
            web_ui.update(
                # Basic
                step_current=iteration,
                epoch_num=1 + iteration // 1000,
                step_max=100000,
                epoch_step_current=iteration % 1000,
                epoch_step_total=1000,
                
                # Losses
                total_loss_value=total_loss,
                l1_loss_value=l1_loss,
                ms_loss_value=ms_loss,
                gradient_loss_value=grad_loss,
                perceptual_loss_value=0.0,
                
                # Adaptive weights
                l1_weight_current=1.2 + random.uniform(-0.1, 0.1),
                ms_weight_current=0.8 + random.uniform(-0.1, 0.1),
                gradient_weight_current=1.0 + random.uniform(-0.1, 0.1),
                perceptual_weight_current=0.0,
                gradient_clip_val=1.0 + random.uniform(-0.2, 0.2),
                
                # Learning rate
                learning_rate_value=learn_rate,
                lr_phase_name='plateau' if iteration > 5000 else 'warmup',
                
                # Performance
                iteration_duration=0.42,
                vram_usage_gb=7.8 + (iteration % 10) * 0.02,
                adam_momentum_avg=0.015 + random.uniform(-0.005, 0.005),
                
                # Time estimates
                eta_total_formatted=eta_total,
                eta_epoch_formatted=epoch_eta,
                
                # Quality
                quality_lr_value=lr_quality,
                quality_ki_value=ki_quality,
                quality_improvement_value=improvement,
                quality_ki_to_gt_value=min(0.95, ki_quality + 0.05),
                quality_lr_to_gt_value=min(0.80, lr_quality + 0.05),
                validation_loss_value=total_loss * 0.9,
                best_quality_ever=min(0.92, 0.5 + iteration / 10000.0),
                
                # Layer activities
                layer_activity_map=layer_activities,
                
                # Status
                training_active=True,
                validation_running=(iteration % 500 == 0),
                training_paused=False,
                
                # Network (will be auto-detected in real use)
                tensorboard_port=6006
            )
            
            # Check for commands
            cmd = web_ui.poll_commands()
            if cmd == 'validate':
                print(f"  [Step {iteration}] ✅ Web UI triggered validation!")
                # Simulate validation
                time.sleep(2)
            
            # Print progress every 100 iterations
            if iteration % 100 == 0:
                print(f"  Iteration {iteration:5d} | Loss: {total_loss:.6f} | "
                      f"KI Quality: {ki_quality*100:.1f}% | Layers: {len(layer_activities)}")
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\n✅ Demo gestoppt")
        web_ui.terminate()
        print("✅ Web UI heruntergefahren")


if __name__ == '__main__':
    demo_complete_web_ui()
