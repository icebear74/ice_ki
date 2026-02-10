# Dataset File Monitoring - Web UI Feature

## Visual Layout

The Web UI now includes a new "Dataset Files" card that displays:

```
┌──────────────────────────────────────────────────┐
│ 📂 Dataset Files                                 │
├──────────────────────────────────────────────────┤
│                                                  │
│ Training Dataset                                 │
│ ┌──────────────────────────────────────────────┐ │
│ │ Size: 540                          12,453   │ │
│ └──────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────┐  │
│ │ ✨ New files detected: 127                 │  │ (Green indicator)
│ └─────────────────────────────────────────────┘  │
│                                                  │
│ Validation Datasets                              │
│ ┌──────────────────────────────────────────────┐ │
│ │ 720×720                             1,234   │ │
│ ├──────────────────────────────────────────────┤ │
│ │ 540×540                               856   │ │
│ │ ┌───────────────────────────────────────────┐│ │
│ │ │ ✨ +42 new files                        ││ │ (Green pulse)
│ │ └───────────────────────────────────────────┘│ │
│ ├──────────────────────────────────────────────┤ │
│ │ 720×405 (16:9)                        723   │ │
│ └──────────────────────────────────────────────┘ │
│                                                  │
│ ─────────────────────────────────────────────── │
│ Last check: Step 15,200                          │
└──────────────────────────────────────────────────┘
```

## Features

1. **Training Dataset Display**
   - Shows the active size key (720, 540, or 720_169)
   - Displays current file count
   - Green notification when new files are detected

2. **Validation Dataset Display**
   - Separate counts for each size variant:
     * 720×720 (square)
     * 540×540 (square)
     * 720×405 (16:9 aspect ratio)
   - Green pulsing indicator for each size with new files

3. **Auto-Update**
   - Checks every 100 training steps
   - Web UI refreshes every 5 seconds
   - Shows last check step number

## Color Scheme

- Background: Dark gradient (#1e293b to #0f172a)
- Card: Semi-transparent white with blur effect
- Title: Blue (#60a5fa)
- Labels: Light gray (#94a3b8)
- Values: White (#e2e8f0)
- New file indicators: Green (#22c55e) with pulse animation

## User Experience

1. During normal training, file counts are displayed
2. When new files are added to any dataset folder, a green indicator appears
3. The indicator shows the number of new files detected
4. Users can monitor dataset growth without stopping training
5. Information persists across page refreshes
