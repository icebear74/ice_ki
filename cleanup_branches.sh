#!/bin/bash

# Farben für Output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Aktualisiere Remote-Informationen...${NC}"
git fetch --prune

echo -e "\n${YELLOW}Suche nach verwaisten lokalen Branches...${NC}\n"

# Finde alle lokalen Branches, deren Remote-Tracking-Branch nicht mehr existiert
deleted_count=0

for branch in $(git branch -vv | grep ': gone]' | awk '{print $1}'); do
    echo -e "${RED}Lösche Branch: ${NC}$branch"
    git branch -D "$branch"
    ((deleted_count++))
done

if [ $deleted_count -eq 0 ]; then
    echo -e "${GREEN}Keine verwaisten Branches gefunden.  Alles sauber!${NC}"
else
    echo -e "\n${GREEN}✓ $deleted_count Branch(es) wurden gelöscht.${NC}"
fi
