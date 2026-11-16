"""Shared settings for the Member 3 pipeline.

Everything marked PROVISIONAL must be confirmed with the owning teammate
(see INTERFACES.md). Change values here only; do not hard-code them elsewhere.
"""

# --- Sequence shape (PROVISIONAL: confirm with Member 1) ---------------------
FRAMES = 60              # frames per model input
FEATURES = 149           # features per frame
WINDOW_SECONDS = 4.0     # length of the rolling landmark buffer

# --- Buffer behaviour (PROVISIONAL) -------------------------------------------
# Minimum share of the window that must be covered by real frames before a
# sequence is released for inference.
MIN_FILL_RATIO = 0.5
# A frame with a larger share of missing/invalid values is rejected outright.
MAX_BAD_VALUE_RATIO = 0.5
# Replacement for NaN/inf values inside an otherwise acceptable frame.
# (PROVISIONAL: confirm missing-landmark convention with Member 1.)
FILL_VALUE = 0.0

# --- Prediction filtering (PROVISIONAL: tune during Weeks 7-9) ----------------
CONFIDENCE_THRESHOLD = 0.70
STABILITY_COUNT = 3          # consecutive matching predictions required
REPEAT_COOLDOWN_SECONDS = 2.0  # same caption is not re-emitted sooner than this
