// JavaScript port of prediction_filter.py. Must behave exactly like the Python
// version (see parity.test.js). Prediction message (PROVISIONAL):
//   { label: "hello", confidence: 0.93, timestamp: 12.35 }   // timestamp in seconds

import { CONFIDENCE_THRESHOLD, STABILITY_COUNT, REPEAT_COOLDOWN_SECONDS } from './config.js';

export class PredictionError extends Error {}

export function parsePrediction(raw) {
  if (raw === null || typeof raw !== 'object' || Array.isArray(raw)) {
    throw new PredictionError('prediction must be an object');
  }
  for (const key of ['label', 'confidence', 'timestamp']) {
    if (!(key in raw)) throw new PredictionError(`missing field: ${key}`);
  }
  const { label } = raw;
  if (typeof label !== 'string' || !label.trim()) {
    throw new PredictionError('label must be a non-empty string');
  }
  const confidence = toNumber(raw.confidence);
  const timestamp = toNumber(raw.timestamp);
  if (confidence === null || timestamp === null) {
    throw new PredictionError('confidence and timestamp must be numbers');
  }
  if (!Number.isFinite(confidence) || confidence < 0 || confidence > 1) {
    throw new PredictionError('confidence must be a finite number in [0, 1]');
  }
  if (!Number.isFinite(timestamp)) throw new PredictionError('timestamp must be finite');
  return { label: label.trim(), confidence, timestamp };
}

// Accepts numbers and numeric strings (like Python's float()); everything else -> null.
function toNumber(v) {
  if (typeof v === 'number') return v;
  if (typeof v === 'string' && v.trim() !== '') {
    const n = Number(v);
    return Number.isNaN(n) ? null : n;
  }
  return null;
}

export class LabelMap {
  constructor(labels) {
    if (!Array.isArray(labels) || labels.length === 0) throw new Error('label list is empty');
    if (labels.some((l) => typeof l !== 'string' || !l.trim())) {
      throw new Error('labels must be non-empty strings');
    }
    if (new Set(labels).size !== labels.length) throw new Error('labels must be unique');
    this.labels = labels.slice();
  }

  get length() {
    return this.labels.length;
  }

  labelFor(index) {
    if (!Number.isInteger(index) || index < 0 || index >= this.labels.length) {
      throw new PredictionError(`class index ${index} out of range`);
    }
    return this.labels[index];
  }
}

// probs: array-like of softmax probabilities (e.g. Float32Array from tfjs .dataSync()).
export function probsToPrediction(probs, labelMap, timestamp) {
  if (probs == null || typeof probs.length !== 'number' || probs.length !== labelMap.length) {
    throw new PredictionError(`expected ${labelMap.length} class scores`);
  }
  let sum = 0;
  let best = 0;
  for (let i = 0; i < probs.length; i++) {
    const p = probs[i];
    if (typeof p !== 'number' || !Number.isFinite(p)) {
      throw new PredictionError('model output contains NaN, infinity or non-numbers');
    }
    if (p < 0 || p > 1 + 1e-6) throw new PredictionError('model output is not a probability distribution');
    sum += p;
    if (p > probs[best]) best = i;
  }
  if (Math.abs(sum - 1) > 1e-3) throw new PredictionError('model output is not a probability distribution');
  return parsePrediction({
    label: labelMap.labelFor(best),
    confidence: Math.min(probs[best], 1),
    timestamp,
  });
}

export class PredictionFilter {
  constructor({
    confidenceThreshold = CONFIDENCE_THRESHOLD,
    stabilityCount = STABILITY_COUNT,
    repeatCooldown = REPEAT_COOLDOWN_SECONDS,
    ignoreLabels = [],
  } = {}) {
    if (stabilityCount < 1) throw new Error('stabilityCount must be >= 1');
    this.confidenceThreshold = confidenceThreshold;
    this.stabilityCount = stabilityCount;
    this.repeatCooldown = repeatCooldown;
    this.ignoreLabels = new Set(ignoreLabels);
    this.reset();
  }

  reset() {
    this._candidate = null;
    this._count = 0;
    this._emittedThisStreak = false;
    this._lastLabel = null;
    this._lastEmitTs = null;
    this.invalidCount = 0;
  }

  _breakStreak() {
    this._candidate = null;
    this._count = 0;
    this._emittedThisStreak = false;
  }

  // Returns caption text to show, or null. Never throws on bad input.
  update(prediction) {
    if (prediction === null || prediction === undefined) {
      this._breakStreak();
      return null;
    }
    let p;
    try {
      p = parsePrediction(prediction);
    } catch (e) {
      if (!(e instanceof PredictionError)) throw e;
      this.invalidCount++;
      this._breakStreak();
      return null;
    }

    if (p.confidence < this.confidenceThreshold || this.ignoreLabels.has(p.label)) {
      this._breakStreak();
      return null;
    }

    if (p.label === this._candidate) {
      this._count++;
    } else {
      this._candidate = p.label;
      this._count = 1;
      this._emittedThisStreak = false;
    }

    if (this._count < this.stabilityCount || this._emittedThisStreak) return null;

    this._emittedThisStreak = true;
    if (
      p.label === this._lastLabel
      && this._lastEmitTs !== null
      && p.timestamp - this._lastEmitTs < this.repeatCooldown
    ) {
      return null;
    }
    this._lastLabel = p.label;
    this._lastEmitTs = p.timestamp;
    return p.label;
  }
}
