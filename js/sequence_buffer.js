// JavaScript port of validation.sanitize_frame + sequence_buffer.RollingBuffer.
// Must behave exactly like the Python version (see parity.test.js).
//
// Timestamps are seconds on any monotonic clock, e.g. performance.now() / 1000.
// getSequence() returns a flat Float32Array of length frames*features, ready for
//   tf.tensor(seq, [1, FRAMES, FEATURES])
// or null while the buffer is not ready.

import {
  FRAMES, FEATURES, WINDOW_SECONDS, MIN_FILL_RATIO, MAX_BAD_VALUE_RATIO, FILL_VALUE,
} from './config.js';

export function sanitizeFrame(
  frame,
  features = FEATURES,
  maxBadRatio = MAX_BAD_VALUE_RATIO,
  fillValue = FILL_VALUE,
) {
  if (frame == null || typeof frame.length !== 'number' || typeof frame === 'string') {
    return { ok: false, frame: null, badValues: 0, reason: 'frame is not an array' };
  }
  if (frame.length !== features) {
    return {
      ok: false, frame: null, badValues: 0,
      reason: `expected ${features} features, got ${frame.length}`,
    };
  }
  const out = new Float32Array(features);
  let bad = 0;
  for (let i = 0; i < features; i++) {
    const v = frame[i];
    if (typeof v !== 'number') {
      return { ok: false, frame: null, badValues: 0, reason: 'frame is not numeric' };
    }
    if (Number.isFinite(v)) {
      out[i] = v;
    } else {
      out[i] = fillValue;
      bad++;
    }
  }
  if (bad / features > maxBadRatio) {
    return {
      ok: false, frame: null, badValues: bad,
      reason: `${bad}/${features} values are NaN/inf`,
    };
  }
  return { ok: true, frame: out, badValues: bad, reason: '' };
}

// Index of the first element >= target (numpy searchsorted, side='left').
function lowerBound(ts, target) {
  let lo = 0;
  let hi = ts.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (ts[mid] < target) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

export class RollingBuffer {
  constructor({
    windowSeconds = WINDOW_SECONDS,
    frames = FRAMES,
    features = FEATURES,
    minFillRatio = MIN_FILL_RATIO,
    maxBadRatio = MAX_BAD_VALUE_RATIO,
    fillValue = FILL_VALUE,
  } = {}) {
    if (frames < 2) throw new Error('frames must be at least 2');
    if (!(windowSeconds > 0)) throw new Error('windowSeconds must be positive');
    this.window = windowSeconds;
    this.frames = frames;
    this.features = features;
    this.minFillRatio = minFillRatio;
    this.maxBadRatio = maxBadRatio;
    this.fillValue = fillValue;
    this._ts = [];
    this._data = [];
    this.rejected = 0;
  }

  // Returns true if accepted, false (and counts it) if rejected.
  push(frame, timestamp) {
    if (typeof timestamp !== 'number' || !Number.isFinite(timestamp)) {
      this.rejected++;
      return false;
    }
    const n = this._ts.length;
    if (n && timestamp < this._ts[n - 1]) {
      this.rejected++;
      return false;
    }
    const r = sanitizeFrame(frame, this.features, this.maxBadRatio, this.fillValue);
    if (!r.ok) {
      this.rejected++;
      return false;
    }
    this._ts.push(timestamp);
    this._data.push(r.frame);
    const cutoff = timestamp - this.window;
    let drop = 0;
    while (drop < this._ts.length && this._ts[drop] < cutoff) drop++;
    if (drop) {
      this._ts.splice(0, drop);
      this._data.splice(0, drop);
    }
    return true;
  }

  clear() {
    this._ts = [];
    this._data = [];
  }

  get length() {
    return this._ts.length;
  }

  get span() {
    const n = this._ts.length;
    return n < 2 ? 0 : this._ts[n - 1] - this._ts[0];
  }

  isReady() {
    return this._ts.length >= 2 && this.span >= this.minFillRatio * this.window;
  }

  getSequence() {
    if (!this.isReady()) return null;
    const ts = this._ts;
    const n = ts.length;
    const tEnd = ts[n - 1];
    const step = this.window / (this.frames - 1);
    const out = new Float32Array(this.frames * this.features);
    for (let i = 0; i < this.frames; i++) {
      const target = i === this.frames - 1 ? tEnd : (tEnd - this.window) + i * step;
      let idx;
      if (n === 1) {
        idx = 0;
      } else {
        const right = Math.min(Math.max(lowerBound(ts, target), 1), n - 1);
        const left = right - 1;
        idx = (target - ts[left]) <= (ts[right] - target) ? left : right;
      }
      out.set(this._data[idx], i * this.features);
    }
    return out;
  }
}
