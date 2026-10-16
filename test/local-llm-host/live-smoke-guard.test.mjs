// test/live/engine-smoke.mjs downloads a runner and a model and starts processes: it must refuse unless PWORKER_LIVE_ENGINE=1, and
// `npm test` must not run it.
import '../home.mjs';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import { scratch } from './fixtures/engine.mjs';

const ROOT = fileURLToPath(new URL('../..', import.meta.url));
const SMOKE = path.join(ROOT, 'test', 'live', 'engine-smoke.mjs');

function smoke(args, env) {
    const { NODE_TEST_CONTEXT: _context, PWORKER_LIVE_ENGINE: _live, ...parent } = process.env;
    return spawnSync(process.execPath, [SMOKE, ...args], { encoding: 'utf8', env: { ...parent, ...env }, timeout: 20_000 });
}

test('without PWORKER_LIVE_ENGINE=1 the smoke script refuses, with exit 2, and creates nothing', (t) => {
    const out = path.join(scratch(t), 'out');
    const result = smoke(['--out', out], {});
    assert.equal(result.status, 2);
    assert.match(result.stderr, /Set PWORKER_LIVE_ENGINE=1 to run it/);
    assert.equal(fs.existsSync(out), false, 'no output folder, no home, no process');
    const other = smoke(['--out', out], { PWORKER_LIVE_ENGINE: '0' });
    assert.equal(other.status, 2, 'only the value 1 enables it');
});

test('a bad option or probe is refused before anything starts, even with the variable set', (t) => {
    const out = path.join(scratch(t), 'out');
    for (const args of [['--out', out, '--probe', 'nonsense'], ['--out', out, '--bogus'], ['--probe', 'ready-kill']]) {
        const result = smoke(args, { PWORKER_LIVE_ENGINE: '1' });
        assert.equal(result.status, 2, args.join(' '));
    }
    assert.equal(fs.existsSync(out), false);
});

test('npm test does not run the live scripts', () => {
    const script = JSON.parse(fs.readFileSync(path.join(ROOT, 'package.json'), 'utf8')).scripts.test;
    assert.ok(!/live/.test(script), script);
    assert.ok(fs.existsSync(SMOKE));
});
