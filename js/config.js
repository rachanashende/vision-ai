// Mirrors config.py. Keep the two files in sync (values marked PROVISIONAL
// must be confirmed with the owning teammate; see INTERFACES.md).

export const FRAMES = 60;            // frames per model input
export const FEATURES = 149;         // features per frame
export const WINDOW_SECONDS = 4.0;   // rolling buffer length

export const MIN_FILL_RATIO = 0.5;
export const MAX_BAD_VALUE_RATIO = 0.5;
export const FILL_VALUE = 0.0;

export const CONFIDENCE_THRESHOLD = 0.7;
export const STABILITY_COUNT = 3;
export const REPEAT_COOLDOWN_SECONDS = 2.0;
