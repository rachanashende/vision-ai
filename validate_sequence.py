
import numpy as np

FRAMES = 60
FEATURES = 149

# Temporary sample data for testing
sequence = np.zeros((FRAMES, FEATURES), dtype=np.float32)

print("Shape:", sequence.shape)
print("Data type:", sequence.dtype)
print("Contains NaN:", np.isnan(sequence).any())
print("Contains infinity:", np.isinf(sequence).any())

assert sequence.shape == (60, 149)
assert sequence.dtype == np.float32
assert np.isfinite(sequence).all()

print("Basic validation passed!")
