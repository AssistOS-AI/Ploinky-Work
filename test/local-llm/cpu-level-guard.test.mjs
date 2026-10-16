// The cpu profile's memory guard on a platform whose pressure reading is the kernel's level (an Intel Mac: `pressureScale: 'level'`
// in darwinPlatform): it is guarded like the metal profile (level 4 stops at once, three elevated samples in a row with the
// available memory below the floor stop, the unreadable memory stops without naming /proc), while the cpu guard of a platform
// without the member (Linux) is the one it always was. Fakes only: no process starts and no system tool runs.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { darwinPlatform } from '../../lib/local-llm/darwinPlatform.mjs';
import { loadSeedCatalog } from '../../lib/local-llm/controller/catalog.mjs';
import { cpuFloorBytes } from '../../lib/local-llm/controller/profiles.mjs';
import { linuxPlatform } from '../../lib/local-llm/host.mjs';
import { SEED_CATALOG_FILE } from './fixtures/seedCatalog.mjs';
import { createController } from './fixtures/host.mjs';

const GIB = 1024 * 1024 * 1024;
const seed = loadSeedCatalog(SEED_CATALOG_FILE);
const tiny = seed.find((model) => model.id === 'qwen2.5-0.5b-instruct-q4_k_m');
const FLOOR = cpuFloorBytes(16 * GIB);

async function waitFor(check, what) {
    for (let i = 0; i < 600; i += 1) {
        if (await check()) return;
        await new Promise((resolve) => setTimeout(resolve, 5));
    }
    assert.fail(`Timed out waiting for ${what}`);
}

/** A cpu-profile controller on `platform` whose memory and pressure readers answer from `state`. */
function harness(t, platform, state) {
    const dataDir = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), 'local-llm-cpu-level-'));
    const handles = [];
    const controller = createController({
        dataDir, hostOverrides: { platform }, env: { PATH: '/usr/bin' }, seedCatalog: seed, imageContract: null, hostArch: 'x64', fileExists: () => true,
        sharedModelsRoot: null, profile: 'cpu', detectRunner: () => ({ installed: true, version: 'b11295', reason: null }),
        snapshot: async () => ({ gpu: { available: false }, memory: { totalBytes: 16 * GIB, availableBytes: 9 * GIB, pressureLevel: 1 },
            disk: { freeBytes: 200 * GIB, totalBytes: 500 * GIB }, cpus: 8, cores: 4 }),
        inspect: async () => ({ state: 'absent', bytes: 0 }),
        download: async () => ({ status: 'complete', path: path.join(dataDir, 'fixture.gguf'), bytesTransferred: 0 }),
        startRunner({ command, args, env }) {
            let running = true; let settle; let killed = false;
            const handle = { command, args, env, pid: 4242, exited: new Promise((resolve) => { settle = resolve; }),
                get running() { return running; }, get killed() { return killed; },
                async stop() { running = false; settle({ code: 0, signal: 'SIGTERM' }); return handle.exited; },
                async kill() { killed = true; return handle.stop(); } };
            handles.push(handle);
            return handle;
        },
        fetchImpl: async () => ({ ok: true, status: 200, json: async () => ({}) }), pollMs: 1,
        readMemory: () => state.memory, readPressure: () => { state.samples += 1; return state.pressure(); }, metalGuardMs: 3, unifiedGuardMs: 3,
    });
    t.after(async () => { await controller.drain(); fs.rmSync(dataDir, { recursive: true, force: true }); });
    return { controller, handles };
}

const level = () => ({ memory: { totalBytes: 16 * GIB, availableBytes: 9 * GIB }, pressure: () => 1, samples: 0 });

async function start(h, requestId) {
    await h.controller.run({ modelId: tiny.id, runnerId: 'llama.cpp', requestId });
    await waitFor(() => h.controller.state.deployment?.phase === 'ready' || h.controller.state.deployment?.phase === 'error', 'ready');
    assert.equal(h.controller.state.deployment.phase, 'ready', JSON.stringify(h.controller.state.deployment));
}

test('the darwin platform says its pressure is a level, and Linux says nothing', () => {
    assert.equal(darwinPlatform().pressureScale, 'level');
    assert.equal(linuxPlatform().pressureScale, undefined);
});

test('cpu on a level platform: pressure level 4 stops the runner at once and the error says why', async (t) => {
    const state = level();
    const h = harness(t, darwinPlatform(), state);
    await start(h, 'cpu-level-000001');
    assert.equal((await h.controller.status()).profile, 'cpu');
    state.pressure = () => 4;
    await waitFor(() => h.controller.state.deployment.phase === 'error', 'the guard stop');
    assert.equal(h.handles[0].killed, true);
    assert.match(h.controller.state.deployment.error, /macOS memory pressure is critical \(level 4\)/);
});

test('cpu on a level platform: elevated pressure with low memory stops after three samples in a row, and not before', async (t) => {
    const state = level();
    const h = harness(t, darwinPlatform(), state);
    await start(h, 'cpu-level-000002');
    // Level 2 with enough memory is no breach; low memory at the normal level is no breach either.
    state.pressure = () => 2;
    state.memory = { totalBytes: 16 * GIB, availableBytes: FLOOR + 1 };
    const before = state.samples;
    await waitFor(() => state.samples >= before + 12, 'twelve clear samples');
    assert.equal(h.handles[0].running, true);
    state.pressure = () => 1;
    state.memory = { totalBytes: 16 * GIB, availableBytes: FLOOR - 1 };
    const normal = state.samples;
    await waitFor(() => state.samples >= normal + 12, 'twelve samples at the normal level');
    assert.equal(h.handles[0].running, true);
    // Level 2 with low memory, three in a row.
    state.pressure = () => 2;
    await waitFor(() => h.controller.state.deployment.phase === 'error', 'the guard stop');
    assert.match(h.controller.state.deployment.error, /macOS memory pressure \(level 2\) for 3 samples in a row/);
    assert.equal(h.handles[0].killed, true);
});

test('cpu on a level platform: memory that cannot be read stops the runner at once, without naming /proc', async (t) => {
    const state = level();
    const h = harness(t, darwinPlatform(), state);
    await start(h, 'cpu-level-000003');
    state.memory = { totalBytes: 16 * GIB, availableBytes: null };
    await waitFor(() => h.controller.state.deployment.phase === 'error', 'the guard stop');
    assert.equal(h.controller.state.deployment.error, 'stopped: host memory cannot be read, so the runner cannot be watched');
});

test('cpu on a platform without the member (Linux) keeps its own guard: a level of 4 is a PSI share of 4 % and stops nothing, memory below the floor stops', async (t) => {
    const state = level();
    state.pressure = () => 4;
    const h = harness(t, linuxPlatform(), state);
    await start(h, 'cpu-level-000004');
    const before = state.samples;
    await waitFor(() => state.samples >= before + 12, 'twelve samples');
    assert.equal(h.handles[0].running, true, 'the level guard is not applied to a PSI reading');
    state.memory = { totalBytes: 16 * GIB, availableBytes: FLOOR - 1 };
    await waitFor(() => h.controller.state.deployment.phase === 'error', 'the guard stop');
    assert.match(h.controller.state.deployment.error, /^stopped: host memory below the floor/);
    assert.doesNotMatch(h.controller.state.deployment.error, /macOS/);
});
