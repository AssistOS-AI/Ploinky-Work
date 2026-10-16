// The metal profile (DS005): admission against the captured Mac and against fabricated ones, the launch, the pure guard
// verdict and the guard inside a controller. The constants are provisional, so the expected figures are written out here
// and a change to a constant must change this file too. The captured Mac's figures come from the macOS readers
// (lib/local-llm-darwin) fed with the output captured on a 16 GiB M1; no system tool runs.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import { admit } from '../../lib/local-llm/controller/admission.mjs';
import { loadSeedCatalog } from '../../lib/local-llm/controller/catalog.mjs';
import { metalGuardVerdict } from '../../lib/local-llm/controller/metalGuard.mjs';
import { METAL, PROFILES, cpuFloorBytes, cpuHostReserveBytes } from '../../lib/local-llm/controller/profiles.mjs';
import { catalogProfileOf } from '../../lib/local-llm/controller/vulkan.mjs';
import { readDarwinMemory, readDarwinPressure } from '../../lib/local-llm-darwin/darwin.mjs';
import { SEED_CATALOG_FILE } from './fixtures/seedCatalog.mjs';
import { createController, getRunner } from './fixtures/host.mjs';

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;
const seed = loadSeedCatalog(SEED_CATALOG_FILE);
const tiny = seed.find((model) => model.id === 'qwen2.5-0.5b-instruct-q4_k_m');
const large = seed.find((model) => model.id === 'qwen3-4b-instruct-2507-q4_k_m');
const runner = getRunner('llama.cpp');
const fixture = (name) => fs.readFileSync(new URL(`../local-llm-darwin/fixtures/${name}`, import.meta.url), 'utf8');
const MTL0_TOTAL = 12_712_935_424;

// ------------------------------------------------------------------ the captured Mac

// The captured sysctl and vm_stat output, as the tools would print it, with the given keys changed.
function capturedExec({ vmStat = fixture('vm_stat.txt'), keys = {} } = {}) {
    const lines = `${fixture('sysctl.txt')}${fixture('swapusage.txt')}`.split('\n').filter(Boolean)
        .map((line) => [line.slice(0, line.indexOf(':')), line]);
    const known = new Map(lines.map(([key, line]) => [key, Object.hasOwn(keys, key) ? `${key}: ${keys[key]}` : line]));
    return (file, args) => {
        if (file === '/usr/bin/vm_stat') return vmStat;
        return `${args.filter((key) => known.has(key)).map((key) => known.get(key)).join('\n')}\n`;
    };
}

const metalGpu = (overrides = {}) => ({ available: true, backend: 'metal', name: 'Apple M1', memoryModel: 'unified', deviceCapBytes: MTL0_TOTAL, ...overrides });
const snapshotOf = (memory, gpu = metalGpu()) => ({ gpu, memory, disk: { freeBytes: 200 * GIB, totalBytes: 500 * GIB }, cpus: 8, cores: 4 });
const roomy = (overrides = {}) => ({ totalBytes: 16 * GIB, availableBytes: 9 * GIB, freeBytes: 2 * GIB, swapFreeBytes: GIB, cachedBytes: null, pressureLevel: 1, ...overrides });

function verdict({ model = tiny, memory = roomy(), gpu = metalGpu(), params = {}, definition = runner, snapshot = snapshotOf(memory, gpu) } = {}) {
    return admit({
        runner: definition, model, source: model.sources.gguf, snapshot, profile: 'metal',
        params: definition.normalizeParams(params, { model, profile: 'metal' }),
    });
}

test('metal-profile: the captured 16 GiB Mac is refused for now (a busy Mac, pressure level 2), not for ever', () => {
    const memory = readDarwinMemory({ exec: capturedExec() });
    assert.equal(memory.totalBytes, 17_179_869_184);
    assert.equal(memory.availableBytes, 2_889_121_792);
    assert.equal(memory.pressureLevel, 2);
    const result = verdict({ memory });
    assert.equal(result.status, 'insufficient-now');
    assert.equal(result.reasonCode, 'metal_memory_now');
    assert.match(result.reason, /available now/);
});

test('metal-profile: a tiny model is admitted on a quiet Mac, with the need written out', () => {
    const result = verdict();
    assert.equal(result.status, 'ok', result.reason);
    // ctx 4096, one slot: the model's KV bytes per token, the compute term without flash attention (2146 MiB x 4096 / 16384), 512 MiB runtime, 256 MiB prompt cache.
    const kv = (tiny.memory?.kvBytesPerToken ?? 65536) * 4096 + (tiny.memory?.fixedKvBytes || 0);
    assert.equal(result.estimate.kvBytes, kv);
    assert.equal(result.estimate.computeBytes, 562_561_024);
    assert.equal(result.estimate.runtimeBytes, 512 * MIB);
    assert.equal(result.estimate.cacheRamBytes, 256 * MIB);
    assert.equal(result.estimate.weightsBytes, tiny.sources.gguf.size);
    assert.equal(result.estimate.needBytes, tiny.sources.gguf.size + kv + 562_561_024 + 512 * MIB + 256 * MIB);
    assert.equal(result.estimate.deviceCapBytes, MTL0_TOTAL);
    assert.equal(result.estimate.floorBytes, cpuFloorBytes(16 * GIB));
    assert.equal(result.estimate.hostReserveBytes, cpuHostReserveBytes(16 * GIB));
    assert.equal(result.estimate.measured, false);
    assert.match(result.warnings[0], /provisional/);
});

test('metal-profile: a need above the device limit less 512 MiB, or above the memory less the host reserve, is incompatible', () => {
    // 1 GiB of device: usable 512 MiB.
    const small = verdict({ gpu: metalGpu({ deviceCapBytes: GIB }) });
    assert.equal(small.status, 'incompatible');
    assert.equal(small.reasonCode, 'metal_capacity');
    // The device limit less the margin is the bound: the smallest cap that admits is the need plus 512 MiB.
    const need = verdict().estimate.needBytes;
    assert.equal(verdict({ gpu: metalGpu({ deviceCapBytes: need + METAL.deviceMarginBytes }) }).status, 'ok');
    assert.equal(verdict({ gpu: metalGpu({ deviceCapBytes: need + METAL.deviceMarginBytes - 1 }) }).status, 'incompatible');
    // The memory less the host reserve is the other bound: a 3 GiB Mac keeps 1.5 GiB for the host.
    const bigDevice = metalGpu({ deviceCapBytes: 64 * GIB });
    const huge = verdict({ model: large, gpu: bigDevice, memory: roomy({ totalBytes: 3 * GIB, availableBytes: 3 * GIB }), params: { ctxSize: 32768 } });
    assert.equal(huge.status, 'incompatible');
    assert.equal(huge.reasonCode, 'metal_capacity');
    assert.equal(verdict({ gpu: bigDevice, memory: roomy({ totalBytes: need + cpuHostReserveBytes(need + 3 * GIB), availableBytes: 64 * GIB }) }).status, 'ok');
});

test('metal-profile: the KV cache counts every slot, so more slots can turn ok into incompatible', () => {
    const one = verdict({ model: large, gpu: metalGpu({ deviceCapBytes: 24 * GIB }), memory: roomy({ totalBytes: 32 * GIB, availableBytes: 28 * GIB }) });
    const four = verdict({ model: large, gpu: metalGpu({ deviceCapBytes: 24 * GIB }), memory: roomy({ totalBytes: 32 * GIB, availableBytes: 28 * GIB }), params: { parallel: 4 } });
    assert.equal(four.estimate.kvBytes, one.estimate.kvBytes * 4 - 3 * (large.memory?.fixedKvBytes || 0));
});

test('metal-profile: pressure level 2 or 4, or an unknown level, gives insufficient-now; level 1 does not', () => {
    for (const pressureLevel of [2, 4, null, undefined, 3, 0]) {
        const result = verdict({ memory: roomy({ pressureLevel }) });
        assert.equal(result.status, 'insufficient-now', String(pressureLevel));
        assert.equal(result.reasonCode, 'metal_pressure', String(pressureLevel));
    }
    assert.equal(verdict({ memory: roomy({ pressureLevel: 1 }) }).status, 'ok');
    // Little available memory is insufficient-now at any pressure: the need must leave the floor free.
    const need = verdict().estimate.needBytes;
    const floor = cpuFloorBytes(16 * GIB);
    assert.equal(verdict({ memory: roomy({ availableBytes: need + floor }) }).status, 'ok');
    assert.equal(verdict({ memory: roomy({ availableBytes: need + floor - 1 }) }).status, 'insufficient-now');
});

test('metal-profile: a missing probe, no device and unreadable memory all refuse', () => {
    for (const deviceCapBytes of [null, undefined, 0, -1, Number.NaN, '12 GiB']) {
        const result = verdict({ gpu: metalGpu({ deviceCapBytes }) });
        assert.equal(result.status, 'incompatible', String(deviceCapBytes));
        assert.equal(result.reasonCode, 'metal_device_unknown');
    }
    const absent = verdict({ gpu: { available: false, backend: 'metal', reason: 'llama-server reported no Metal device.' } });
    assert.equal(absent.status, 'incompatible');
    assert.equal(absent.reasonCode, 'metal_no_device');
    assert.equal(verdict({ gpu: metalGpu({ backend: 'cuda' }) }).status, 'incompatible');
    assert.equal(verdict({ snapshot: { memory: roomy(), disk: { freeBytes: GIB } } }).status, 'incompatible');
    for (const memory of [{}, { totalBytes: null, availableBytes: null }, roomy({ availableBytes: null }), roomy({ totalBytes: Number.NaN })]) {
        const result = verdict({ memory });
        assert.equal(result.status, 'incompatible');
        assert.equal(result.reasonCode, 'metal_memory_unreadable');
    }
});

test('metal-profile: garbage vm_stat output refuses admission and nothing crashes', () => {
    const memory = readDarwinMemory({ exec: capturedExec({ vmStat: 'Mach Virtual Memory Statistics: ???\nPages free: lots.\n' }) });
    assert.equal(memory.availableBytes, null);
    const result = verdict({ memory });
    assert.equal(result.status, 'incompatible');
    assert.equal(result.reasonCode, 'metal_memory_unreadable');
    assert.equal(admit({ runner, model: tiny, source: tiny.sources.gguf, snapshot: snapshotOf({}), profile: 'metal', params: runner.normalizeParams({}, { model: tiny, profile: 'metal' }) }).status, 'incompatible');
});

test('metal-profile: a disk too small for the download is insufficient-now, and a runner without a metal policy is incompatible', () => {
    const disk = admit({ runner, model: tiny, source: tiny.sources.gguf, remainingDownloadBytes: 10 * GIB, profile: 'metal',
        snapshot: { ...snapshotOf(roomy()), disk: { freeBytes: GIB, totalBytes: 100 * GIB } }, params: runner.normalizeParams({}, { model: tiny, profile: 'metal' }) });
    assert.equal(disk.status, 'insufficient-now');
    assert.match(disk.reason, /free disk/);
    for (const id of ['ik_llama.cpp', 'ollama', 'vllm', 'tabbyapi', 'lmstudio', 'llama.cpp-cpu', 'llama.cpp-vulkan']) {
        const other = getRunner(id);
        const result = admit({ runner: other, model: tiny, source: tiny.sources.gguf, snapshot: snapshotOf(roomy()), profile: 'metal', params: {} });
        assert.equal(result.status, 'incompatible', id);
    }
});

test('metal-profile: the pressure level reader gives 1, 2 or 4 and the catalog maps metal to cpu', () => {
    assert.equal(readDarwinPressure({ exec: capturedExec() }), 2);
    assert.equal(readDarwinPressure({ exec: capturedExec({ keys: { 'kern.memorystatus_vm_pressure_level': '1' } }) }), 1);
    assert.equal(catalogProfileOf('metal'), 'cpu');
    assert.equal(PROFILES.includes('metal'), false);
});

// ------------------------------------------------------------------ the launch

test('metal-profile: the launch passes the Metal flags, the performance cores and no environment', () => {
    const launch = runner.buildLaunch({
        artifactPath: '/models/m.gguf', params: { ctxSize: 2048 }, port: 18080, apiKey: 'k'.repeat(40), model: tiny, profile: 'metal', runnerDir: '/runners/llama.cpp',
    });
    const tail = launch.args.slice(launch.args.indexOf('-c'));
    assert.equal(tail[0], '-c');
    assert.deepEqual(tail.slice(0, 2), ['-c', '2048']);
    assert.deepEqual(tail.slice(2, 10), ['-ngl', '999', '-fa', 'auto', '-ctk', 'f16', '-ctv', 'f16']);
    assert.equal(tail[10], '--threads');
    assert.match(tail[11], /^\d+$/);
    assert.deepEqual(tail.slice(12), ['-np', '1', '--cache-ram', '256']);
    assert.deepEqual(launch.env, {});
    for (const forbidden of ['--device', '--n-gpu-layers', '--flash-attn', '--batch-size', '--load-mode']) assert.equal(launch.args.includes(forbidden), false, forbidden);
    // More than one slot shares one KV cache; an explicit thread count wins.
    const two = runner.buildLaunch({ artifactPath: '/models/m.gguf', params: { parallel: 2, threads: 3 }, port: 18080, apiKey: 'k'.repeat(40), model: tiny, profile: 'metal', runnerDir: '/r' });
    assert.ok(two.args.includes('--kv-unified'));
    assert.equal(two.args[two.args.indexOf('--threads') + 1], '3');
    assert.equal(two.args[two.args.indexOf('-np') + 1], '2');
});

test('metal-profile: the default threads are the performance cores the platform counts, and parameters outside the schema are refused', async () => {
    const { createRunners } = await import('../../lib/local-llm/runners/index.mjs');
    const { defineHost } = await import('../../lib/local-llm/host.mjs');
    const { darwinPlatform } = await import('../../lib/local-llm/darwinPlatform.mjs');
    const platform = darwinPlatform({ exec: (file, args) => (args[1] === 'hw.perflevel0.physicalcpu' ? '4\n' : '10\n') });
    const host = defineHost({ dataDir: '/x/data', runRoot: '/x/run', platform });
    const metal = createRunners({ host })['llama.cpp'];
    const launch = metal.buildLaunch({ artifactPath: '/m.gguf', params: {}, port: 18080, apiKey: 'k'.repeat(40), model: tiny, profile: 'metal', runnerDir: '/r' });
    assert.equal(launch.args[launch.args.indexOf('--threads') + 1], '4');
    // A platform that cannot count them leaves the runner to count the cores itself.
    const unknown = defineHost({ dataDir: '/x/data', runRoot: '/x/run', platform: darwinPlatform({ exec: () => { throw new Error('no sysctl'); } }) });
    const fallback = createRunners({ host: unknown })['llama.cpp'].buildLaunch({ artifactPath: '/m.gguf', params: {}, port: 18080, apiKey: 'k'.repeat(40), model: tiny, profile: 'metal', runnerDir: '/r' });
    assert.match(fallback.args[fallback.args.indexOf('--threads') + 1], /^\d+$/);
    for (const params of [{ nGpuLayers: 10 }, { loadMode: 'none' }, { parallel: 5 }, { ctxSize: 100 }, { flashAttn: 'off' }]) {
        assert.throws(() => metal.normalizeParams(params, { model: tiny, profile: 'metal' }), { name: 'ParamError' }, JSON.stringify(params));
    }
});

// ------------------------------------------------------------------ the guard verdict

const FLOOR = cpuFloorBytes(16 * GIB);
const LOW = FLOOR - 1;
const HIGH = FLOOR + 1;

test('metal-profile: the guard verdict stops at level 4 at once', () => {
    const step = metalGuardVerdict({ pressureLevel: 4, available: 8 * GIB, floorBytes: FLOOR, breaches: 0 });
    assert.notEqual(step.stop, null);
    assert.match(step.stop, /^stopped: macOS memory pressure is critical \(level 4\) with 8\.0 GiB available$/);
});

test('metal-profile: the guard verdict stops after 3 breaches in a row, not after 2, and starts again from zero when one sample is clear', () => {
    let state = { breaches: 0, stop: null };
    const sample = (pressureLevel, available) => { state = metalGuardVerdict({ pressureLevel, available, floorBytes: FLOOR, breaches: state.breaches }); return state; };
    assert.equal(sample(2, LOW).stop, null);
    assert.equal(sample(2, LOW).stop, null);
    assert.equal(state.breaches, 2);
    const third = sample(2, LOW);
    assert.match(third.stop, /^stopped: macOS memory pressure \(level 2\) for 3 samples in a row with 1\.\d GiB available, below the 1\.\d GiB floor$/);
    // A clear sample in between starts the count again.
    state = { breaches: 0, stop: null };
    sample(2, LOW); sample(2, LOW);
    assert.equal(sample(1, LOW).breaches, 0, 'normal pressure');
    sample(2, LOW); sample(2, LOW);
    assert.equal(sample(2, HIGH).breaches, 0, 'enough memory');
    assert.equal(sample(2, LOW).stop, null);
    // Elevated pressure with enough memory, and a low memory at normal pressure, are not breaches by themselves.
    for (const [level, available] of [[2, HIGH], [2, FLOOR], [1, LOW], [1, 0]]) assert.deepEqual(metalGuardVerdict({ pressureLevel: level, available, floorBytes: FLOOR, breaches: 2 }), { breaches: 0, stop: null }, `${level} ${available}`);
});

test('metal-profile: a pressure level that cannot be read counts as elevated', () => {
    for (const pressureLevel of [null, undefined, Number.NaN, 3]) {
        assert.equal(metalGuardVerdict({ pressureLevel, available: LOW, floorBytes: FLOOR, breaches: 1 }).breaches, 2, String(pressureLevel));
        assert.match(metalGuardVerdict({ pressureLevel, available: LOW, floorBytes: FLOOR, breaches: METAL.guardBreaches - 1 }).stop, /level (unknown|3)/);
        assert.equal(metalGuardVerdict({ pressureLevel, available: HIGH, floorBytes: FLOOR, breaches: 1 }).breaches, 0);
    }
});

// ------------------------------------------------------------------ the guard inside a controller

function harness(t, { memory = { totalBytes: 16 * GIB, availableBytes: 9 * GIB }, pressure = () => 1 } = {}) {
    const dataDir = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), 'local-llm-metal-'));
    const handles = [];
    const readings = { pressure: 0 };
    const controller = createController({
        dataDir, env: { PATH: '/usr/bin', SECRET_SENTINEL: 'private-value' }, seedCatalog: seed, imageContract: null, hostArch: 'arm64', fileExists: () => true,
        sharedModelsRoot: null, profile: 'metal', detectRunner: () => ({ installed: true, version: 'b11295', reason: null }),
        snapshot: async () => snapshotOf({ ...roomy(), totalBytes: memory.totalBytes, availableBytes: memory.availableBytes }),
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
        readMemory: () => memory, readPressure: () => { readings.pressure += 1; return pressure(readings.pressure); }, metalGuardMs: 3,
    });
    t.after(async () => { await controller.drain(); fs.rmSync(dataDir, { recursive: true, force: true }); });
    return { controller, handles, memory, readings };
}
async function waitFor(check, what = 'the controller state') {
    for (let i = 0; i < 600; i += 1) {
        if (await check()) return;
        await new Promise((resolve) => setTimeout(resolve, 5));
    }
    assert.fail(`Timed out waiting for ${what}`);
}
const start = async (h, requestId) => {
    await h.controller.run({ modelId: tiny.id, runnerId: 'llama.cpp', requestId });
    await waitFor(() => h.controller.state.deployment?.phase === 'ready', 'ready');
};

test('metal-profile: a run on the metal profile launches the Metal flags and watches the runner with the pressure level', async (t) => {
    const h = harness(t);
    await start(h, 'metal-run-000001');
    const [handle] = h.handles;
    assert.equal(handle.args[handle.args.indexOf('-ngl') + 1], '999');
    for (const name of ['LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES', 'VK_DRIVER_FILES', 'SECRET_SENTINEL']) assert.equal(Object.hasOwn(handle.env, name), false, name);
    await waitFor(async () => (await h.controller.status()).memoryGuard?.samples > 2, 'guard samples');
    const status = await h.controller.status();
    assert.equal(status.profile, 'metal');
    assert.equal(status.memoryGuard.maxPressureLevel, 1);
    assert.equal(status.memoryGuard.floorBytes, cpuFloorBytes(16 * GIB));
    assert.equal(handle.running, true);
});

test('metal-profile: pressure level 4 kills the runner group at once and the error says why', async (t) => {
    let level = 1;
    const h = harness(t, { pressure: () => level });
    await start(h, 'metal-run-000002');
    level = 4;
    await waitFor(() => h.controller.state.deployment.phase === 'error', 'the guard stop');
    assert.equal(h.handles[0].killed, true);
    assert.match(h.controller.state.deployment.error, /macOS memory pressure is critical \(level 4\)/);
    assert.equal(h.handles.length, 1);
});

// What the fake pressure reader answers: clear (level 1), elevated (level 2), or two elevated samples and one clear, again and again.
function pressureModes() {
    const mode = { current: 'clear' };
    let step = 0;
    return { mode, read: () => (mode.current === 'clear' ? 1 : mode.current === 'elevated' ? 2 : [2, 2, 1][step++ % 3]) };
}

test('metal-profile: two elevated samples in a row with low memory never stop the runner', async (t) => {
    const floor = cpuFloorBytes(16 * GIB);
    const { mode, read } = pressureModes();
    const h = harness(t, { pressure: read });
    await start(h, 'metal-run-000003');
    h.memory.availableBytes = floor - 1;
    mode.current = 'cycle';
    const before = h.readings.pressure;
    await waitFor(() => h.readings.pressure >= before + 15, 'fifteen guard samples');
    assert.equal(h.handles[0].running, true);
    assert.equal(h.controller.state.deployment.phase, 'ready');
});

test('metal-profile: elevated pressure with low memory for three samples in a row kills the runner, and the error says so', async (t) => {
    const floor = cpuFloorBytes(16 * GIB);
    const { mode, read } = pressureModes();
    const h = harness(t, { pressure: read });
    await start(h, 'metal-run-000004');
    h.memory.availableBytes = floor - 1;
    const before = h.readings.pressure;
    mode.current = 'elevated';
    await waitFor(() => h.controller.state.deployment.phase === 'error', 'the guard stop');
    assert.equal(h.handles[0].killed, true);
    assert.match(h.controller.state.deployment.error, /macOS memory pressure \(level 2\) for 3 samples in a row/);
    assert.ok(h.readings.pressure - before >= 3, 'at least three samples were read');
});

test('metal-profile: memory that cannot be read stops the runner at once, without naming /proc', async (t) => {
    const h = harness(t);
    await start(h, 'metal-run-000005');
    h.memory.availableBytes = null;
    await waitFor(() => h.controller.state.deployment.phase === 'error', 'the guard stop');
    assert.equal(h.handles[0].killed, true);
    assert.equal(h.controller.state.deployment.error, 'stopped: host memory cannot be read, so the runner cannot be watched');
});

test('metal-profile: pressure level 2 or 4 refuses a run before anything is downloaded or started', async (t) => {
    for (const level of [2, 4]) {
        const dataDir = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), 'local-llm-metal-refuse-'));
        let downloads = 0;
        const handles = [];
        const controller = createController({
            dataDir, env: { PATH: '/usr/bin' }, seedCatalog: seed, imageContract: null, hostArch: 'arm64', fileExists: () => true, sharedModelsRoot: null, profile: 'metal',
            detectRunner: () => ({ installed: true, version: 'b11295', reason: null }),
            snapshot: async () => snapshotOf(roomy({ pressureLevel: level })), inspect: async () => ({ state: 'absent', bytes: 0 }),
            download: async () => { downloads += 1; return { status: 'complete', path: path.join(dataDir, 'x.gguf'), bytesTransferred: 0 }; },
            startRunner: () => { handles.push(1); return null; },
        });
        t.after(async () => { await controller.drain(); fs.rmSync(dataDir, { recursive: true, force: true }); });
        await assert.rejects(controller.run({ modelId: tiny.id, runnerId: 'llama.cpp', requestId: `metal-refuse-0000${level}` }),
            (error) => error.code === 'admission_insufficient_now' && error.details.admission.reasonCode === 'metal_pressure');
        assert.equal(downloads, 0);
        assert.equal(handles.length, 0);
    }
});

// ------------------------------------------------------------------ the catalog

const source = { type: 'huggingface', repo: 'test/model', file: 'model.gguf', commit: 'a'.repeat(40), revision: 'main', size: 400000000, sha256: 'b'.repeat(64) };
const metalModel = () => ({ id: 'metal-model', sources: { gguf: source }, recommended: { metal: { 'llama.cpp': { ctxSize: 2048 } } }, validated: { metal: { 'llama.cpp': 'measured on an M1' } } });

test('metal-profile: the catalog accepts metal metadata for llama.cpp only, from a trusted seed only, and v3 refuses it', async () => {
    const { validateModel } = await import('../../lib/local-llm/controller/catalog.mjs');
    const { exportModelMetadata } = await import('../../lib/local-llm/controller/venusMetadata.mjs');
    const entry = validateModel(metalModel(), { seed: true });
    assert.deepEqual(entry.profiles, ['dedicated', 'unified', 'cpu'], 'metal metadata does not change which profiles list the model');
    assert.deepEqual(entry.recommended.metal, { 'llama.cpp': { ctxSize: 2048 } });
    // A label is a certification, which a user entry may not carry; a recommendation is allowed.
    assert.throws(() => validateModel(metalModel()), /trusted seed/);
    assert.doesNotThrow(() => validateModel({ ...metalModel(), validated: undefined }));
    for (const field of ['recommended', 'validated']) {
        const wrong = metalModel();
        wrong[field].metal = { 'llama.cpp-vulkan': wrong[field].metal['llama.cpp'] };
        assert.throws(() => validateModel(wrong, { seed: true }), { code: 'invalid_model' }, field);
    }
    assert.throws(() => validateModel({ ...metalModel(), profiles: ['metal'] }, { seed: true }), /profiles/);
    // A v3 catalog does not know the profile, and the lossy export names what it drops.
    assert.throws(() => validateModel(metalModel(), { seed: true, schema: 'local-llm.catalog/v3' }), /unknown profile/);
    assert.throws(() => exportModelMetadata([metalModel()], { seed: true, targetSchema: 'local-llm.catalog/v3' }), { code: 'catalog_downgrade_lossy' });
    const out = exportModelMetadata([metalModel()], { seed: true, targetSchema: 'local-llm.catalog/v3', dropVenusMetadata: true });
    assert.deepEqual(out.dropped, [{ modelId: 'metal-model', field: 'recommended.metal' }, { modelId: 'metal-model', field: 'validated.metal' }]);
    assert.deepEqual(out.models[0].recommended, {});
});

test('metal-profile: the schema file lists metal under recommended and validated with the llama.cpp runner only', () => {
    const schema = JSON.parse(fs.readFileSync(new URL('../../lib/local-llm/catalog/schema.json', import.meta.url), 'utf8'));
    const found = [];
    const walk = (node) => {
        if (node && typeof node === 'object') {
            if (node.recommended?.properties && node.validated?.properties) found.push(node);
            Object.values(node).forEach(walk);
        }
    };
    walk(schema);
    assert.equal(found.length, 1);
    for (const field of ['recommended', 'validated']) {
        const metal = found[0][field].properties.metal;
        assert.equal(metal.additionalProperties, false, field);
        assert.deepEqual(Object.keys(metal.properties), ['llama.cpp'], field);
    }
});

// ------------------------------------------------------------------ the path a Mac takes: no executable, the lock as the image lock

test('metal-profile: on a Mac host (no executables, the arm64 lock as the image lock) llama.cpp is installable, the others refuse, and the launch command is the runnable copy', async (t) => {
    const { defineHost } = await import('../../lib/local-llm/host.mjs');
    const { darwinPlatform } = await import('../../lib/local-llm/darwinPlatform.mjs');
    const { loadRunnerLocks } = await import('../../lib/local-llm/controller/runnerLock.mjs');
    const { createRunners } = await import('../../lib/local-llm/runners/index.mjs');
    const dataDir = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), 'local-llm-metal-host-'));
    t.after(() => fs.rmSync(dataDir, { recursive: true, force: true }));
    const lockFile = fileURLToPath(new URL('../../lib/local-llm-darwin/locks/runners.lock.darwin-arm64.json', import.meta.url));
    const host = defineHost({ dataDir, runRoot: path.join(dataDir, 'run'), platform: darwinPlatform(), imageRunnerLockFile: lockFile });
    assert.deepEqual(host.executables, {});
    let downloads = 0;
    const controller = createController({
        host, seedCatalog: seed, hostArch: 'arm64', profile: 'metal', env: { PATH: '/usr/bin' },
        runnerLocks: loadRunnerLocks({ image: lockFile, agent: null }),
        snapshot: async () => snapshotOf(roomy()),
        download: async () => { downloads += 1; return { status: 'complete', path: path.join(dataDir, 'x.gguf'), bytesTransferred: 0 }; },
    });
    t.after(() => controller.drain());
    const view = await controller.overview();
    assert.equal(view.profile, 'metal');
    const byId = Object.fromEntries(view.runners.map((entry) => [entry.id, entry]));
    // llama.cpp: a parameter schema for the profile, installable from the lock at b11295, not installed yet.
    assert.notEqual(byId['llama.cpp'].paramSchema, null);
    assert.equal(byId['llama.cpp'].install.version, 'b11295');
    assert.equal(byId['llama.cpp'].install.totalBytes, 11_817_829);
    assert.equal(byId['llama.cpp'].installed, false);
    // The summary still shows the adapter's own version beside the lock's: observed, and a follow-up for the host.
    assert.equal(byId['llama.cpp'].pinnedVersion, 'b11159');
    // Every other runner says it is not on the Apple GPU and offers no schema for it.
    for (const id of ['ik_llama.cpp', 'ollama', 'vllm', 'tabbyapi', 'lmstudio']) {
        assert.equal(byId[id].paramSchema, null, id);
        assert.match(byId[id].profileUnsupportedReason, /not available on the Apple GPU/, id);
    }
    assert.equal(Object.hasOwn(byId, 'llama.cpp-cpu'), false, 'the cpu build is not offered on the metal profile');
    assert.equal(Object.hasOwn(byId, 'llama.cpp-vulkan'), false);
    // A Run of another runner, or of llama.cpp before it is installed, is refused before anything is downloaded.
    await assert.rejects(controller.run({ modelId: tiny.id, runnerId: 'ik_llama.cpp', requestId: 'metal-host-000001' }), { code: 'runner_unavailable', message: /not available on the Apple GPU/ });
    await assert.rejects(controller.run({ modelId: tiny.id, runnerId: 'llama.cpp', requestId: 'metal-host-000002' }), { code: 'runner_not_installed' });
    assert.equal(downloads, 0);
    assert.equal(controller.state.deployment, null);
    // The launch runs the runnable copy of the lock's install, not an executable of the host.
    const metal = createRunners({ host })['llama.cpp'];
    const launch = metal.buildLaunch({ artifactPath: '/models/m.gguf', params: {}, port: 18080, apiKey: 'k'.repeat(40), model: tiny, profile: 'metal', runnerDir: '/runners/llama.cpp/b11295' });
    assert.equal(launch.command, '/runners/llama.cpp/b11295/llama-server');
});
