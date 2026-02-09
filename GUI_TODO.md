# GUI Implementation - TODO

## Current Status

The generator now has:
- ✅ Video duration scanning with progress bar
- ✅ Proportional distribution table
- ✅ Basic progress display

## Still Missing

Full Rich GUI from `make_dataset_multi.py` (line 724):
- Live updating display
- Overall progress bar
- Per-category progress bars with ETA
- Current video progress bar
- Disk usage display
- Live controls display

## Implementation Plan

1. **Add build_gui_layout() method** (from make_dataset_multi.py line 724)
   - Header panel with current video
   - Overall progress section
   - Current video progress bar
   - Category progress bars
   - Disk usage section
   - Controls section

2. **Add Live display loop** (from make_dataset_multi.py line 870)
   - Update display every 0.5 seconds
   - Refresh GUI layout
   - Show live statistics

3. **Add helper methods**
   - `_calculate_bar_widths()` - Dynamic bar sizing
   - `_should_update_display()` - Throttle updates
   - `_build_simple_status()` - Fallback without Rich

## Files to Reference

- `dataset_generator_v2/make_dataset_multi.py` - Lines 724-870
- Full GUI implementation available there

## Recommendation

For now, the generator has:
- ✅ All critical functionality (stacking, aspect ratio, duration analysis)
- ⚠️ Basic progress display (not full GUI)

Full GUI can be added in next iteration if needed.

Users can run `make_dataset_multi.py` if they need the full GUI experience right now.
