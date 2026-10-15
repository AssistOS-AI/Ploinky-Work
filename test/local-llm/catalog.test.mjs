// Catalog schema v3: model sources are keyed by weight format (gguf, ollama,
// hf, exl3), so runners that read the same format share one download;
// `recommended` and `validated` are keyed by hardware profile, then by runner;
// `profiles` says where a model is offered; `unified.envelope` is the unified
// profile's measured envelope (DS005). Earlier schemas are not migrated.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { WEIGHT_FORMATS, loadSeedCatalog, mergeCatalog, validateModel } from '../../lib/local-llm/controller/catalog.mjs';
import { STATE_VERSION, createStateStore } from '../../lib/local-llm/controller/stateStore.mjs';

const GGUF = Object.freeze({
    type: 'huggingface', repo: 'Qwen/Qwen3-0.6B-GGUF', file: 'Qwen3-0.6B-Q8_0.gguf', revision: 'main',
    commit: 'a'.repeat(40), size: 639446688, sha256: 'b'.repeat(64),
});
const OLLAMA = Object.freeze({ type: 'ollama', tag: 'qwen3:0.6b' });
const RECTANGLE = Object.freeze({ runner: 'llama.cpp', loadMode: 'dio', maxCtx: 131072, maxParallel: 4, mtp: false, bufferBytes: 17_000_000_000, transientBytes: 2_000_000_000 });

function tempStore(t, content) {
    const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-catalog-'));
    t.after(() => fs.rmSync(dataDir, { recursive: true, force: true }));
    const store = createStateStore({ dataDir });
    if (content !== undefined) {
        fs.mkdirSync(path.dirname(store.file), { recursive: true });
        fs.writeFileSync(store.file, JSON.stringify(content));
    }
    return store;
}

test('an entry is validated per format; runner-keyed sources are refused', () => {
    const model = validateModel({ id: 'qwen-small', sources: { gguf: GGUF, ollama: OLLAMA } });
    assert.deepEqual(Object.keys(model.sources), ['gguf', 'ollama']);
    assert.throws(() => validateModel({ id: 'qwen-small', sources: { 'llama.cpp': GGUF } }),
        { code: 'invalid_model', message: /unknown weight format 'llama\.cpp'/ });
    assert.throws(() => validateModel({ id: 'qwen-small', sources: { gguf: OLLAMA } }), /sources\.gguf\.type must be huggingface/);
    assert.throws(() => validateModel({ id: 'qwen-small', sources: { ollama: GGUF } }), /sources\.ollama\.type must be ollama/);
});

test('profiles default to every profile; recommended and validated are keyed by profile, then runner', () => {
    // `validated` is a measurement record: only a trusted seed entry carries one.
    const model = validateModel({
        id: 'qwen-small', sources: { gguf: GGUF },
        recommended: { dedicated: { 'llama.cpp': { ctxSize: 8192 } }, unified: { 'llama.cpp': { ctxSize: 32768 } } },
        validated: { unified: { 'llama.cpp': 'DGX Spark, b11159' } },
    }, { seed: true });
    assert.deepEqual(model.profiles, ['dedicated', 'unified', 'cpu']);
    assert.deepEqual(model.recommended.unified['llama.cpp'], { ctxSize: 32768 });
    assert.equal(model.validated.unified['llama.cpp'], 'DGX Spark, b11159');
    assert.equal(model.unified, null);
    assert.deepEqual(validateModel({ id: 'x-only', sources: { gguf: GGUF }, profiles: ['unified'] }).profiles, ['unified']);
    assert.deepEqual(validateModel({ id: 'cpu-only', sources: { gguf: GGUF }, profiles: ['cpu'] }).profiles, ['cpu']);
    for (const profiles of [[], ['gpu'], ['unified', 'unified'], ['cpu', 'cpu'], 'unified']) {
        assert.throws(() => validateModel({ id: 'bad-profiles', sources: { gguf: GGUF }, profiles }),
            { message: /profiles must list one or more of dedicated, unified, cpu/ });
    }
    // `recommended` and `validated` are keyed by cpu too, and an unknown profile is still refused.
    const cpuKeyed = validateModel({ id: 'cpu-keyed', sources: { gguf: GGUF }, recommended: { cpu: { 'llama.cpp': { ctxSize: 4096 } } },
        validated: { cpu: { 'llama.cpp': 'M1 Podman machine' } } }, { seed: true });
    assert.deepEqual(cpuKeyed.recommended.cpu['llama.cpp'], { ctxSize: 4096 });
    assert.throws(() => validateModel({ id: 'gpu-keyed', sources: { gguf: GGUF }, recommended: { gpu: { 'llama.cpp': {} } } }),
        /recommended has an unknown profile 'gpu'/);
    // A v2 entry (recommended keyed by runner) is not valid v3.
    assert.throws(() => validateModel({ id: 'v2-entry', sources: { gguf: GGUF }, recommended: { 'llama.cpp': { ctxSize: 8192 } } }),
        /recommended has an unknown profile 'llama\.cpp'/);
    assert.throws(() => validateModel({ id: 'v2-entry', sources: { gguf: GGUF }, validated: { unified: { 'llama.cpp': 42 } } }, { seed: true }),
        /validated\.unified\.llama\.cpp must be plain text/);
});

test('the unified envelope is measured rectangles, each with its load mode, buffers and transient', () => {
    const model = validateModel({ id: 'enveloped', sources: { gguf: GGUF }, unified: { envelope: [RECTANGLE, { ...RECTANGLE, maxCtx: 262144, maxParallel: 1, mtp: true, measured: 'L1, 2026-09-28' }] } }, { seed: true });
    assert.equal(model.unified.envelope.length, 2);
    assert.deepEqual(model.unified.envelope[0], { ...RECTANGLE, measured: undefined });
    assert.equal(model.unified.envelope[1].measured, 'L1, 2026-09-28');
    for (const [change, pattern] of [
        [{ bufferBytes: undefined }, /bufferBytes is required/],
        [{ transientBytes: -1 }, /transientBytes must be an integer/],
        [{ maxParallel: 17 }, /maxParallel must be an integer from 1 to 16/],
        [{ mtp: 'yes' }, /mtp must be true or false/],
        [{ runner: '' }, /runner must name a runner/],
        [{ extra: 1 }, /unsupported field 'extra'/],
        // The load mode is part of what was measured: it is required, one mode per rectangle.
        [{ loadMode: undefined }, /loadMode must be the load mode it was measured with \(dio or none\)/],
        [{ loadMode: 'mmap' }, /loadMode must be the load mode it was measured with/],
        [{ loadMode: ['dio', 'none'] }, /loadMode must be the load mode it was measured with/],
    ]) {
        const entry = { ...RECTANGLE, ...change };
        for (const [key, value] of Object.entries(change)) if (value === undefined) delete entry[key];
        assert.throws(() => validateModel({ id: 'enveloped', sources: { gguf: GGUF }, unified: { envelope: [entry] } }, { seed: true }), pattern);
    }
    assert.throws(() => validateModel({ id: 'enveloped', sources: { gguf: GGUF }, unified: { envelope: [] } }, { seed: true }), /1-16 measured rectangles/);
});

test('a registry entry of an earlier schema is kept in the state file but hidden from the catalog', (t) => {
    const v1 = { id: 'user-qwen', displayName: 'Qwen', sources: { 'llama.cpp': GGUF, ollama: OLLAMA } };
    const v2 = { id: 'user-v2', sources: { gguf: GGUF }, recommended: { 'llama.cpp': { ctxSize: 8192 } } };
    const v3 = { id: 'user-v3', sources: { gguf: GGUF }, recommended: { dedicated: { 'llama.cpp': { ctxSize: 8192 } } } };
    const store = tempStore(t, { version: 1, registry: [v1, v2, v3] });
    const state = store.load();
    assert.equal(state.version, STATE_VERSION);
    assert.deepEqual(state.registry, [v1, v2, v3]);
    assert.deepEqual(mergeCatalog([], state.registry).map((model) => model.id), ['user-v3']);
});

test('two user entries with one id keep only the first, so lookups and removal agree', () => {
    const first = { id: 'user-qwen', displayName: 'First', sources: { gguf: GGUF } };
    const second = { id: 'user-qwen', displayName: 'Second', sources: { ollama: OLLAMA } };
    assert.deepEqual(mergeCatalog([], [first, second]).map((model) => model.displayName), ['First']);
});

test('a state file from a newer controller is kept aside, not overwritten silently', (t) => {
    const store = tempStore(t, { version: 99, registry: [{ id: 'x' }] });
    const state = store.load();
    assert.equal(state.version, STATE_VERSION);
    assert.deepEqual(state.registry, []);
    const kept = fs.readdirSync(path.dirname(store.file)).filter((name) => name.startsWith('controller.json.unsupported-'));
    assert.equal(kept.length, 1);
});
