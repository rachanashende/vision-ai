// Replays the inputs recorded by tools/make_parity_fixtures.py (Python) and
// requires identical outputs from the JavaScript implementation.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { RollingBuffer } from './sequence_buffer.js';
import { PredictionFilter } from './prediction_filter.js';

const fixtures = JSON.parse(
  readFileSync(new URL('./fixtures/parity.json', import.meta.url), 'utf8'),
);

// JSON has no NaN, so the generator writes null; turn it back into NaN here.
const unNull = (frame) => frame.map((v) => (v === null ? NaN : v));

for (const c of fixtures.buffers) {
  test(`buffer parity: ${c.name}`, () => {
    const buf = new RollingBuffer(c.params);
    const accepted = c.pushes.map((p) => buf.push(unNull(p.frame), p.t));

    assert.deepEqual(accepted, c.accepted, 'accept/reject decisions differ');
    assert.equal(buf.rejected, c.rejected);
    assert.equal(buf.length, c.length);
    assert.equal(buf.isReady(), c.ready);

    const seq = buf.getSequence();
    if (c.sequence === null) {
      assert.equal(seq, null);
    } else {
      assert.equal(seq.length, c.sequence.length);
      let firstDiff = -1;
      for (let i = 0; i < seq.length; i++) {
        if (seq[i] !== c.sequence[i]) { firstDiff = i; break; }
      }
      assert.equal(firstDiff, -1, `sequence differs at flat index ${firstDiff}`);
    }
  });
}

for (const c of fixtures.filters) {
  test(`filter parity: ${c.name}`, () => {
    const flt = new PredictionFilter(c.params);
    const outputs = c.inputs.map((raw) => flt.update(raw));
    assert.deepEqual(outputs, c.outputs);
    assert.equal(flt.invalidCount, c.invalidCount);
    assert.ok(outputs.some((o) => o !== null), 'fixture should contain captions');
  });
}
