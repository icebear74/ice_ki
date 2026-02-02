#!/bin/bash

# ============================================================
# Quick Start Script - VSR+++ Option B
# ============================================================

GREEN='\033[92m'
CYAN='\033[96m'
YELLOW='\033[93m'
BOLD='\033[1m'
RESET='\033[0m'

echo -e "${BOLD}${CYAN}VSR+++ Quick Start${RESET}"
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}⚠ Virtuelle Umgebung nicht gefunden!${RESET}"
    echo -e "Bitte führen Sie zuerst das Setup aus:"
    echo -e "  ${CYAN}bash setup_env.sh${RESET}"
    echo ""
    exit 1
fi

# Activate environment
echo -e "${CYAN}Aktiviere Umgebung...${RESET}"
source venv/bin/activate

# Check if model file exists
if [ ! -f "model_vsrppp_v2.py" ]; then
    echo -e "${YELLOW}⚠ model_vsrppp_v2.py nicht gefunden!${RESET}"
    exit 1
fi

# Check if train.py exists
if [ ! -f "train.py" ]; then
    echo -e "${YELLOW}⚠ train.py nicht gefunden!${RESET}"
    exit 1
fi

echo -e "${GREEN}✓ Umgebung aktiviert${RESET}"
echo ""
echo -e "${BOLD}Verfügbare Befehle:${RESET}"
echo ""
echo -e "${CYAN}Training:${RESET}"
echo -e "  python train.py"
echo ""
echo -e "${CYAN}TensorBoard (in separatem Terminal):${RESET}"
echo -e "  source venv/bin/activate"
echo -e "  tensorboard --logdir /mnt/data/training/Universal/Mastermodell/Learn/logs"
echo ""
echo -e "${CYAN}Info:${RESET}"
echo -e "  python -c 'from model_vsrppp_v2 import VSRTriplePlus_3x; m = VSRTriplePlus_3x(); print(f\"Model: {sum(p.numel() for p in m.parameters())/1e6:.2f}M params\")'"
echo ""
echo -e "${YELLOW}Um das Training zu starten, führen Sie aus:${RESET}"
echo -e "${BOLD}python train.py${RESET}"
echo ""
