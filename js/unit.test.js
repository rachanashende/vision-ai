import test from 'node:test';
import assert from 'node:assert/strict';
import { FRAMES, FEATURES } from './config.js';
import { RollingBuffer, sanitizeFrame } from './sequence_buffer.js';
import {
  LabelMap, PredictionError, PredictionFilter, parsePrediction, probsToPrediction,
} from './prediction_filter.js';

const frame = (v) => new Float32Array(FEATURES).fill(v);
const P = (label, confidence = 0.9, timestamp = 0) => ({ label, confidence, timestamp });

test('sanitizeFrame accepts arrays and typed arrays, fills stray NaN', () => {
  assert.ok(sanitizeFrame(new Array(FEATURES).fill(1)).ok);
  const f = frame(1);
  f[3] = NaN;
  f[4] = Infinity;
  const r = sanitizeFrame(f);
  assert.ok(r.ok);
  assert.equal(r.badValues, 2);
  assert.equal(r.frame[3], 0);
});

test('sanitizeFrame rejects bad input', () => {
  assert.ok(!sanitizeFrame(null).ok);
  assert.ok(!sanitizeFrame('abc').ok);
  assert.ok(!sanitizeFrame(new Array(FEATURES - 1).fill(0)).ok);
  assert.ok(!sanitizeFrame(new Array(FEATURES).fill(null)).ok); // null is not a number
  assert.ok(!sanitizeFrame(new Float32Array(FEATURES).fill(NaN)).ok);
});

test('buffer: not ready until half the window is covered', () => {
  const buf = new RollingBuffer();
  for (let i = 0; i < 10; i++) buf.push(frame(i), i / 15);
  assert.equal(buf.getSequence(), null);
  for (let i = 10; i < 61; i++) buf.push(frame(i), i / 15);
  const seq = buf.getSequence();
  assert.equal(seq.length, FRAMES * FEATURES);
  assert.ok(seq instanceof Float32Array);
  assert.equal(seq[0], 0);
  assert.equal(seq[(FRAMES - 1) * FEATURES], 60); // latest frame is last
});

test('buffer: evicts frames older than the window', () => {
  const buf = new RollingBuffer();
  for (let i = 0; i < 200; i++) buf.push(frame(i), i / 15);
  assert.ok(buf.span <= 4 + 1e-9);
});

test('buffer: rejects out-of-order and non-finite timestamps', () => {
  const buf = new RollingBuffer();
  assert.ok(buf.push(frame(0), 1));
  assert.ok(!buf.push(frame(0), 0.5));
  assert.ok(!buf.push(frame(0), NaN));
  assert.ok(!buf.push(frame(0), '2'));
  assert.equal(buf.rejected, 3);
});

test('probsToPrediction picks argmax and rejects non-probabilities', () => {
  const lm = new LabelMap(['hello', 'thanks', 'please']);
  const p = probsToPrediction(new Float32Array([0.1, 0.7, 0.2]), lm, 4.2);
  assert.equal(p.label, 'thanks');
  assert.ok(Math.abs(p.confidence - 0.7) < 1e-6);
  assert.throws(() => probsToPrediction([3, 1, -2], lm, 0), PredictionError);
  assert.throws(() => probsToPrediction([0.5, 0.5], lm, 0), PredictionError);
  assert.throws(() => probsToPrediction([0.1, NaN, 0.9], lm, 0), PredictionError);
});

test('parsePrediction rejects malformed messages', () => {
  assert.throws(() => parsePrediction(null), PredictionError);
  assert.throws(() => parsePrediction({ label: 'a' }), PredictionError);
  assert.throws(() => parsePrediction(P('a', 1.5)), PredictionError);
  assert.throws(() => parsePrediction(P('', 0.5)), PredictionError);
});

test('filter: three in a row, once per held sign', () => {
  const f = new PredictionFilter();
  const out = Array.from({ length: 10 }, (_, i) => f.update(P('hello', 0.9, i * 0.35)));
  assert.deepEqual(out.filter(Boolean), ['hello']);
  assert.equal(out[2], 'hello');
});

test('filter: low confidence and malformed input break the streak', () => {
  const f = new PredictionFilter();
  const out = [P('a'), P('a'), P('a', 0.5), P('a'), P('a'), { label: 'a' }, P('a'), null]
    .map((x) => f.update(x));
  assert.ok(out.every((o) => o === null));
  assert.equal(f.invalidCount, 1);
});
