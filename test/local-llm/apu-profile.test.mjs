import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { admit, admitLlamaServer } from '../../lib/local-llm/controller/admission.mjs';
import { loadSeedCatalog } from '../../lib/local-llm/controller/catalog.mjs';
import { SEED_CATALOG_FILE } from './fixtures/seedCatalog.mjs';
import { createController } from './fixtures/host.mjs';
import { readSnapshot } from '../../lib/local-llm/controller/hardware.mjs';
import { APU, cpuFloorBytes, cpuHostReserveBytes, decideProfile } from '../../lib/local-llm/controller/profiles.mjs';
import { createLogBuffer, createReportCollector, parseRunnerReport } from '../../lib/local-llm/controller/runnerProcess.mjs';
import { createStateStore, emptyState } from '../../lib/local-llm/controller/stateStore.mjs';
import { catalogProfileOf, readVulkanGpu, vulkanContractOf } from '../../lib/local-llm/controller/vulkan.mjs';
import { runnerSummary } from '../../lib/local-llm/runners/index.mjs';
import { HOST, HOST_VULKAN, PLOINKY_MESSAGES, getRunner, RUNNERS } from './fixtures/host.mjs';
import { cpuFs, VULKAN_KEYS } from './fixtures/sharedFixtures.mjs';
import { defineLimits } from '../../lib/local-llm/limits.mjs';

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;
const seed = loadSeedCatalog(SEED_CATALOG_FILE);
const model = seed.find((entry) => entry.id === 'qwen2.5-0.5b-instruct-q4_k_m');
const runner = getRunner('llama.cpp-vulkan');
const contract = vulkanContractOf(VULKAN_KEYS, HOST_VULKAN);
const fixture = (name) => fs.readFileSync(new URL(`./fixtures/vulkan/${name}`, import.meta.url), 'utf8');
const gpu = () => ({ available: true, backend: 'vulkan', driver: 'radv', memoryModel: 'shared', name: 'AMD Radeon Graphics',
    totalBytes: 16 * GIB, freeBytes: 16 * GIB, icdFile: contract.vulkan_icd_radv, processes: [] });
const snapshot = () => ({ gpu: gpu(), memory: { totalBytes: 32 * GIB, availableBytes: 24 * GIB }, disk: { freeBytes: 100 * GIB } });
const reportLines = [
    'using device Vulkan0 (AMD Radeon Graphics) (RADV) - 16384 MiB free',
    'offloaded 25/25 layers to GPU',
    'Vulkan0 model buffer size = 325.02 MiB',
    'Vulkan0 KV buffer size = 48.00 MiB',
];

function probe({ summary = fixture('radv-integrated-summary.txt'), devices = fixture('radv-integrated-devices.txt') } = {}) {
    const calls = [];
    return { calls, execFileImpl(command, args, options, callback) {
        calls.push({ command, args, options });
        if (command.endsWith('nvidia-smi')) callback(Object.assign(new Error('absent'), { code: 'ENOENT' }), '', '');
        else callback(null, command === contract.vulkaninfo ? summary : devices, '');
        return null;
    } };
}
async function readApu({ enabled = true, ...options } = {}) {
    const host = probe(options);
    const fake = cpuFs();
    const snap = await readSnapshot({ dataDir: '/tmp', apu: enabled, vulkan: contract, execFileImpl: host.execFileImpl,
        nvidiaSmi: HOST.nvidiaSmi, messages: PLOINKY_MESSAGES,
        env: {}, statfs: async () => ({ bsize: 4096, blocks: 1000000, bavail: 999000 }),
        fsApi: { ...fake, readdirSync: () => ['renderD128'], lstatSync: () => ({ isCharacterDevice: () => true }),
            readFileSync(file) { return fake.readFileSync(file); } } });
    return { snap, calls: host.calls };
}
function verdict({ snap = snapshot(), definition = runner, chosenModel = model, params = {}, profile = 'apu' } = {}) {
    return admit({ runner: definition, model: chosenModel, source: chosenModel.sources.gguf,
        params: definition.normalizeParams ? definition.normalizeParams(params, { model: chosenModel, profile }) : params,
        snapshot: snap, profile, messages: PLOINKY_MESSAGES });
}

test('T2-T4: enabled integrated RADV selects apu; discrete RADV and Venus keep their profiles', async () => {
    const { snap } = await readApu();
    assert.equal(snap.gpu.available, true);
    assert.equal(snap.gpu.memoryModel, 'shared');
    assert.equal(decideProfile({ snapshot: snap }).profile, 'apu');
    for (const [kind, expected] of [['radv', 'vulkan'], ['venus', 'venus']]) {
        const { snap: other } = await readApu({ summary: fixture(`${kind}-summary.txt`), devices: fixture(`${kind}-devices.txt`) });
        assert.equal(decideProfile({ snapshot: other }).profile, expected);
    }
});
test('T2: optional summary metadata survives on APU only', async () => {
    const summary = fixture('radv-integrated-summary.txt') + '    deviceID = 0x1638\n    driverInfo = Mesa 25.3.6\n    apiVersion = 1.4.0\n';
    const { snap } = await readApu({ summary });
    assert.equal(snap.gpu.deviceID, '0x1638');
    assert.equal(snap.gpu.driverInfo, 'Mesa 25.3.6');
    assert.equal(snap.gpu.apiVersion, '1.4.0');
});
test('T6: APU routing calls only admitApu and refuses other driver/memory pairs', () => {
    const calls = [];
    const definition = { supported: true, displayName: 'spy', admitApu: () => { calls.push('apu'); return { status: 'ok' }; },
        admitVulkan: () => calls.push('vulkan'), admitVenus: () => calls.push('venus') };
    assert.equal(verdict({ definition }).status, 'ok');
    assert.deepEqual(calls, ['apu']);
    for (const change of [{ driver: 'venus' }, { memoryModel: 'dedicated' }]) {
        const snap = snapshot(); Object.assign(snap.gpu, change);
        assert.match(verdict({ definition, snap }).reason, /restart local-llm/);
    }
    assert.deepEqual(calls, ['apu']);
});
test('T6: every other runner refuses APU; catalog uses CPU metadata without APU recommendations', () => {
    assert.equal(catalogProfileOf('apu'), 'cpu');
    for (const definition of Object.values(RUNNERS).filter((entry) => entry !== runner)) {
        const result = admit({ runner: definition, model, source: model.sources.gguf, params: {}, snapshot: snapshot(), profile: 'apu' });
        assert.equal(result.status, 'incompatible', definition.id);
        assert.match(result.reason, /Vulkan build/);
        assert.match(runnerSummary(definition, 'apu').profileUnsupportedReason, /Vulkan build/);
    }
    assert.deepEqual(runner.normalizeParams({}, { model, profile: 'apu' }), { ctxSize: 4096, parallel: 1 });
});
test('T7: APU schema fixes all memory flags, Vulkan0 and RADV environment', () => {
    const launch = runner.buildLaunch({ model, profile: 'apu', params: { parallel: 2 }, artifactPath: '/data/test.gguf',
        port: 18086, apiKey: 'x'.repeat(32), icdFile: contract.vulkan_icd_radv });
    for (const [key, value] of [['--device', 'Vulkan0'], ['-ngl', '999'], ['--fit', 'off'], ['--load-mode', 'none'],
        ['--flash-attn', 'auto'], ['-ctk', 'f16'], ['-ctv', 'f16'], ['-b', '512'], ['-ub', '512'], ['--cache-ram', '256']]) {
        assert.equal(launch.args[launch.args.indexOf(key) + 1], value, key);
    }
    assert.ok(launch.args.includes('--kv-unified'));
    assert.deepEqual(launch.env, { VK_DRIVER_FILES: contract.vulkan_icd_radv, VK_LOADER_LAYERS_DISABLE: '~all~' });
    for (const params of [{ nGpuLayers: 0 }, { loadMode: 'auto' }, { threads: 1 }, { ctxSize: 511 }, { parallel: 5 }]) {
        assert.throws(() => runner.normalizeParams(params, { model, profile: 'apu' }));
    }
    assert.throws(() => runner.buildLaunch({ model, profile: 'apu', artifactPath: '/data/test.gguf', port: 18086,
        apiKey: 'x'.repeat(32), icdFile: contract.vulkan_icd_venus }), /ICD file/);
});
test('T11: two-pool admission accepts a 6 GiB model which the single APU pool refuses now', () => {
    const chosenModel = { ...model, memory: { ...model.memory, layers: 24 }, sources: { gguf: { ...model.sources.gguf, size: 6 * GIB } } };
    const snap = snapshot(); snap.memory.availableBytes = 10 * GIB; snap.gpu.freeBytes = 10 * GIB;
    const params = { ctxSize: 4096, parallel: 1, nGpuLayers: 999, nCpuMoe: 0, batchSize: 512, ubatchSize: 512, flashAttn: 'off' };
    assert.equal(admitLlamaServer({ model: chosenModel, source: chosenModel.sources.gguf, params, gpu: snap.gpu, memory: snap.memory }).status, 'ok');
    const result = verdict({ chosenModel, snap });
    assert.equal(result.status, 'insufficient-now'); assert.equal(result.estimate.constraint, 'C1');
    assert.ok(result.estimate.needLoadPeak > 12 * GIB);
});
test('T11: peak counts device bytes once, staging once and CPU-scaled reserves', () => {
    const result = verdict(); const e = result.estimate;
    assert.equal(result.status, 'ok');
    assert.equal(e.needLoadPeak, e.deviceBytes + e.stagingBytes + e.runtimeBytes + e.cacheRamBytes);
    assert.equal(e.floorBytes, cpuFloorBytes(snapshot().memory.totalBytes));
    assert.equal(e.hostReserveBytes, cpuHostReserveBytes(snapshot().memory.totalBytes));
    assert.equal(e.measured, false); assert.equal(APU.minModelBufferFraction, null);
});
test('T11-T12: physical total, device total, device free and disk constraints fail closed', () => {
    for (const [change, status, constraint] of [
        [(snap) => { snap.memory.totalBytes = snap.memory.availableBytes = 2 * GIB; }, 'incompatible', 'C1'],
        [(snap) => { snap.gpu.totalBytes = snap.gpu.freeBytes = 128 * MIB; }, 'incompatible', 'C2'],
        [(snap) => { snap.gpu.freeBytes = 0; }, 'insufficient-now', 'C2'],
        [(snap) => { snap.memory = {}; }, 'incompatible', 'C1'],
        [(snap) => { snap.memory.availableBytes = 40 * GIB; }, 'incompatible', 'C1'],
    ]) {
        const snap = snapshot(); change(snap); const result = verdict({ snap });
        assert.equal(result.status, status); assert.equal(result.estimate.constraint, constraint);
    }
    const snap = snapshot(); snap.disk.freeBytes = 100;
    const result = admit({ runner, model, source: model.sources.gguf, params: { ctxSize: 4096, parallel: 1 }, snapshot: snap,
        profile: 'apu', remainingDownloadBytes: 101 });
    assert.equal(result.status, 'insufficient-now'); assert.equal(result.estimate.constraint, 'C4');
});
test('T12: contradictory and unsafe Vulkan byte counts never become APU capacity', async () => {
    for (const [total, free, expected] of [['16384', '16385', 'unusable'], ['0', '0', 'unusable'], ['9007199254740992', '0', 'unreadable']]) {
        const host = probe({ devices: `Available devices:\n  Vulkan0: AMD Radeon Graphics (${total} MiB, ${free} MiB free)\n` });
        assert.equal((await readVulkanGpu({ contract, apuEnabled: true, execFileImpl: host.execFileImpl })).state, expected);
    }
});

function harness(t, { lines = reportLines, ready = true, env = {}, initialState = null, lookup = undefined,
    installer = undefined, downloadImpl = null, killImpl = null, monotonicNow = null, activityCounterLimit = undefined,
    onSnapshot = null, limits = undefined } = {}) {
    const root = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), 'local-llm-apu-'));
    const memory = { totalBytes: 32 * GIB, availableBytes: 24 * GIB };
    const time = { monotonic: 0, wall: Date.now() }; const handles = []; const calls = [];
    let snap = snapshot(); let downloads = 0; let responseReady = ready;
    const store = createStateStore({ dataDir: root });
    if (initialState) store.save(initialState);
    const controller = createController({ dataDir: root, env: { PATH: '/usr/bin', LOCAL_LLM_AMD_APU: 'experimental', SECRET_SENTINEL: 'private-value', ...env },
        ...(limits ? { limits } : {}),
        stateStore: store, seedCatalog: seed, imageContract: VULKAN_KEYS, hostArch: 'x64', fileExists: () => true,
        sharedModelsRoot: null, detectRunner: () => ({ installed: true, version: 'b11159', reason: null }),
        ...(lookup ? { lookup } : {}), ...(installer ? { installer } : {}),
        ...(activityCounterLimit === undefined ? {} : { apuActivityCounterLimit: activityCounterLimit }),
        readHardwareSnapshot: async (options) => { calls.push(options); await onSnapshot?.(); return snap; },
        inspect: async () => ({ state: 'absent', bytes: 0 }),
        download: async (...args) => { downloads += 1; return downloadImpl ? downloadImpl(...args)
            : { status: 'complete', path: path.join(root, 'test.gguf'), bytesTransferred: 0 }; },
        startRunner({ command, args, env: runnerEnv, log }) {
            let running = true; let settle; let killed = false;
            const handle = { pid: 4242, command, args, env: runnerEnv, log, exited: new Promise((resolve) => { settle = resolve; }),
                get running() { return running; }, get killed() { return killed; },
                async stop() { running = false; settle({ code: 0, signal: 'SIGTERM' }); return handle.exited; },
                async kill() { killed = true; return killImpl ? killImpl(handle) : handle.stop(); } };
            handles.push(handle);
            for (const line of lines) log.append('stderr', line);
            return handle;
        },
        fetchImpl: async () => ({ ok: responseReady, status: responseReady ? 200 : 503, json: async () => ({}) }),
        pollMs: 2, unifiedGuardMs: 5, readMemory: () => memory, readPressure: () => 0,
        monotonicNow: monotonicNow || (() => time.monotonic), now: () => new Date(time.wall),
    });
    t.after(async () => { await controller.drain(); fs.rmSync(root, { recursive: true, force: true }); });
    return { controller, handles, memory, time, calls, get downloads() { return downloads; },
        setSnapshot(value) { snap = value; }, ready() { responseReady = true; } };
}
async function waitFor(check) {
    for (let i = 0; i < 500; i += 1) {
        if (await check()) return;
        await new Promise((resolve) => setTimeout(resolve, 5));
    }
    assert.fail('Timed out waiting for APU controller state');
}
const run = (h, params = {}, requestId = 'apu-run-000001') => h.controller.run({ modelId: model.id, runnerId: runner.id, params, requestId });

test('controller passes the exact experimental switch to the default snapshot reader', async (t) => {
    for (const [value, expected] of [['experimental', true], ['true', false], ['', false], ['Experimental', false]]) {
        const h = harness(t, { env: { LOCAL_LLM_AMD_APU: value } }); await h.controller.overview();
        assert.equal(h.calls[0].apu, expected);
    }
});
test('T11: an insufficient shared pool refuses Run before downloader or runner', async (t) => {
    const h = harness(t); const snap = snapshot(); snap.memory.availableBytes = 512 * MIB; h.setSnapshot(snap);
    await assert.rejects(run(h), (error) => error.code === 'admission_insufficient_now' && error.details.admission.estimate.constraint === 'C1');
    assert.equal(h.downloads, 0); assert.equal(h.handles.length, 0); assert.equal(h.controller.state.deployment, null);
});
test('T13: guard is pooled, kills below admission floor and never relaunches CPU', async (t) => {
    const h = harness(t); await run(h); await waitFor(() => h.controller.state.deployment?.phase === 'ready');
    await waitFor(async () => (await h.controller.status()).memoryGuard?.samples > 0);
    h.memory.availableBytes = 0;
    await waitFor(() => h.controller.state.deployment.phase === 'error');
    assert.equal(h.handles[0].killed, true); assert.equal(h.handles.length, 1);
    assert.match(h.controller.state.deployment.error, /below the floor/);
    assert.equal(h.controller.state.deployment.profile, 'apu');
});
// The guard hooks (slice C of the engine move): the pooled guard samples the limits through guardSample, asks guardBreach, and
// uses its answers. The recording plug-in below answers as the identity unless a test gives it a script.
function guardLimits({ sample = () => null, breach = () => null } = {}) {
    const calls = [];
    const limits = defineLimits({
        guardSample() { calls.push({ name: 'guardSample' }); return sample(); },
        guardBreach(value, context) { calls.push({ name: 'guardBreach', sample: value, context }); return breach(value, context); },
    });
    return { limits, calls, headroom: () => calls.filter((call) => call.name === 'guardBreach' && call.context.check === 'headroom') };
}
test('guard hooks: a watched first sample samples again before every check; the headroom answer is what is compared with the floor', async (t) => {
    let cap = null;
    const g = guardLimits({ sample: () => ({ watched: true }), breach: (value, context) => context.check === 'headroom'
        ? { available: cap ?? context.available } : null });
    const h = harness(t, { limits: g.limits }); await run(h); await waitFor(() => h.controller.state.deployment?.phase === 'ready');
    await waitFor(async () => (await h.controller.status()).memoryGuard?.samples > 1);
    assert.deepEqual(g.calls.slice(0, 5).map((call) => call.name === 'guardBreach' ? `guardBreach:${call.context.check}` : call.name),
        ['guardSample', 'guardSample', 'guardBreach:watchable', 'guardSample', 'guardBreach:headroom']);
    const watchable = g.calls.find((call) => call.name === 'guardBreach' && call.context.check === 'watchable');
    assert.deepEqual(watchable.sample, { watched: true });
    assert.equal(watchable.context.profile, 'apu');
    const first = g.headroom()[0];
    assert.deepEqual(first.sample, { watched: true });
    assert.equal(first.context.memory, h.memory);
    assert.equal(first.context.available, 24 * GIB);
    assert.equal(h.handles[0].running, true);
    cap = 0;
    await waitFor(() => h.controller.state.deployment.phase === 'error');
    assert.match(h.controller.state.deployment.error, /below the floor \(0\.0 GiB available/);
    assert.equal(h.handles[0].killed, true); assert.equal(h.handles.length, 1);
});
test('guard hooks: two consecutive unreadable answers stop the runner with the last message; a readable one in between resets the count', async (t) => {
    const script = ['unreadable', 'ok', 'unreadable', 'unreadable'];
    let index = 0;
    const g = guardLimits({ sample: () => ({ watched: true }), breach: (value, context) => {
        if (context.check !== 'headroom') return null;
        const step = script[Math.min(index, script.length - 1)]; index += 1;
        return step === 'unreadable' ? { unreadable: `stopped: unreadable reading ${index}` } : { available: context.available };
    } });
    const h = harness(t, { limits: g.limits }); await run(h);
    await waitFor(() => h.controller.state.deployment?.phase === 'error');
    assert.equal(g.headroom().length, 4, 'one readable answer in between restarts the count');
    assert.equal(h.controller.state.deployment.error, 'stopped: unreadable reading 4');
    assert.equal(h.handles[0].killed, true); assert.equal(h.handles.length, 1);
});
test('guard hooks: a guard whose first sample is not watched never asks for headroom; a sample that throws is no sample', async (t) => {
    for (const [sample, given] of [[() => ({ watched: false }), { watched: false }], [() => null, null], [() => { throw new Error('unreadable'); }, null]]) {
        const g = guardLimits({ sample, breach: (value, context) => context.check === 'headroom' ? { unreadable: 'must not be asked' } : null });
        const h = harness(t, { limits: g.limits }); await run(h); await waitFor(() => h.controller.state.deployment?.phase === 'ready');
        await waitFor(async () => (await h.controller.status()).memoryGuard?.samples > 2);
        assert.equal(g.headroom().length, 0);
        assert.equal(h.controller.state.deployment.phase, 'ready');
        const watchable = g.calls.find((call) => call.name === 'guardBreach');
        assert.equal(watchable.context.check, 'watchable');
        assert.deepEqual(watchable.sample, given);
        await h.controller.stop();
    }
});
test('guard hooks: the APU guard stops the runner with the message the limits give for a runner they cannot watch', async (t) => {
    const g = guardLimits({ breach: (value, context) => context.check === 'watchable' ? { message: 'stopped: this runner cannot be watched here' } : null });
    const h = harness(t, { limits: g.limits }); await run(h);
    await waitFor(() => h.controller.state.deployment?.phase === 'error');
    assert.equal(h.controller.state.deployment.error, 'stopped: this runner cannot be watched here');
    assert.equal(h.handles[0].killed, true); assert.equal(h.handles.length, 1);
    assert.equal(h.controller.state.deployment.profile, 'apu');
});
test('T13: raise-only runtime floor kills early while admission and dashboard keep their floors', async (t) => {
    const h = harness(t, { ready: false, env: { LOCAL_LLM_APU_GUARD_FLOOR_BYTES: String(33 * GIB) } });
    const baseline = verdict(); await run(h);
    assert.equal(h.controller.state.deployment.admission.estimate.floorBytes, baseline.estimate.floorBytes);
    await waitFor(() => h.controller.state.deployment.phase === 'error');
    const status = await h.controller.status();
    assert.equal(status.memoryGuard.floorBytes, 33 * GIB);
    assert.equal((await h.controller.overview()).limits.floorBytes, baseline.estimate.floorBytes);
    assert.equal(h.handles[0].killed, true);
});
for (const value of ['1', 'broken', '-2', '9007199254740992']) test(`T13: invalid/lower floor ${value} is ignored with an overview warning`, async (t) => {
    const h = harness(t, { env: { LOCAL_LLM_APU_GUARD_FLOOR_BYTES: value } });
    await run(h); await waitFor(() => h.controller.state.deployment.phase === 'ready');
    const view = await h.controller.overview(); assert.match(view.limits.warnings.join(' '), /ignored/);
    const status = await h.controller.status();
    assert.equal(status.memoryGuard.floorBytes, view.limits.floorBytes);
    assert.equal(h.handles[0].running, true);
});
test('T13: monotonic 2.5 s sample gap kills; 10 s wall jump alone does not', async (t) => {
    const h = harness(t); await run(h); await waitFor(() => h.controller.state.deployment.phase === 'ready');
    await waitFor(async () => (await h.controller.status()).memoryGuard?.samples > 0);
    h.time.wall += 10000; const before = (await h.controller.status()).memoryGuard.samples;
    await waitFor(async () => (await h.controller.status()).memoryGuard.samples > before);
    assert.equal(h.handles[0].running, true);
    h.time.monotonic += 2500;
    await waitFor(() => h.controller.state.deployment.phase === 'error');
    assert.match(h.controller.state.deployment.error, /telemetry stale/);
    assert.equal((await h.controller.status()).memoryGuard.maxMonotonicGapMs, 2500);
});
test('T13: unreadable host memory while starting kills immediately', async (t) => {
    const h = harness(t, { ready: false }); await run(h); await waitFor(() => h.handles.length === 1);
    delete h.memory.availableBytes; await waitFor(() => h.controller.state.deployment.phase === 'error');
    assert.equal(h.handles[0].killed, true); assert.match(h.controller.state.deployment.error, /cannot be read/);
});
test('T14: valid Vulkan buffers prove readiness and observed ratio, with minimal runner env', async (t) => {
    const h = harness(t); await run(h); await waitFor(() => h.controller.state.deployment.phase === 'ready');
    const status = await h.controller.status();
    assert.equal(status.runnerReport.modelBufferRatio, 325.02 * MIB / model.sources.gguf.size);
    assert.match(status.runnerReport.modelBufferDenominatorBasis, /full pinned weights/);
    assert.equal(status.apuObservation.deploymentId, status.deployment.id);
    assert.equal(status.apuObservation.requestId, status.deployment.requestId);
    assert.equal(status.apuObservation.pid, 4242);
    assert.equal(status.apuObservation.command, runner.executable);
    assert.deepEqual(Object.keys(h.handles[0].env).sort(), ['PATH', 'HOME', 'LANG', 'CUDA_CACHE_PATH', 'VK_DRIVER_FILES', 'VK_LOADER_LAYERS_DISABLE', 'MESA_SHADER_CACHE_DIR'].sort());
    assert.doesNotMatch(JSON.stringify(status), /private-value|x{32}/);
});
for (const [name, lines] of [
    ['missing KV', reportLines.filter((line) => !line.includes('KV buffer'))],
    ['missing model', reportLines.filter((line) => !line.includes('model buffer'))],
    ['CPU_Mapped only', reportLines.map((line) => line.replaceAll('Vulkan0', 'CPU_Mapped'))],
    ['layer echo only', reportLines.slice(0, 2)],
    ['partial layers', reportLines.map((line) => line.replace('25/25', '0/25'))],
]) test(`T14: ${name} fails offload proof, kills and never falls back`, async (t) => {
    const h = harness(t, { lines }); await run(h); await waitFor(() => h.controller.state.deployment.phase === 'error');
    assert.match(h.controller.state.deployment.error, /gpu_offload_unproven/);
    assert.equal(h.handles[0].killed, true); assert.equal(h.handles.length, 1);
    assert.equal(h.controller.state.deployment.profile, 'apu');
});
test('T14: incremental totals survive 3000 filler lines and reset with each launch', async (t) => {
    const h = harness(t, { lines: [...reportLines, 'Vulkan0 model buffer size = 10.00 MiB', 'Vulkan0 KV buffer size = 2.00 MiB',
        ...Array.from({ length: 3000 }, (_, i) => `filler ${i}`)] });
    await run(h); await waitFor(() => h.controller.state.deployment.phase === 'ready');
    let status = await h.controller.status(); assert.equal(status.runnerReport.modelMiB, 335.02); assert.equal(status.runnerReport.kvMiB, 50);
    assert.ok(status.logs.every((entry) => !entry.line.includes('buffer size')));
    await h.controller.stop(); await run(h, {}, 'apu-second-0001'); await waitFor(() => h.controller.state.deployment.phase === 'ready');
    status = await h.controller.status(); assert.equal(status.runnerReport.modelMiB, 335.02); assert.equal(status.runnerReport.kvMiB, 50);
});
test('T14: collector ignores controller echoes and keeps first device and offload markers', () => {
    const collector = createReportCollector(); const log = createLogBuffer({ limit: 1, onLine: (line) => collector.append(line) });
    for (const line of reportLines) log.append('stderr', line);
    log.append('controller', 'Vulkan0 model buffer size = 100000 MiB');
    log.append('stderr', 'offloaded 0/25 layers to GPU');
    assert.equal(collector.read().modelMiB, 325.02); assert.equal(collector.read().offloaded.layers, 25);
    assert.equal(log.all().length, 1);
});
test('T14: late output from a stopped launch cannot satisfy a later launch proof', async (t) => {
    const h = harness(t, { ready: false, lines: [] });
    await run(h); await waitFor(() => h.handles.length === 1);
    const old = h.handles[0]; await h.controller.stop();
    await run(h, {}, 'apu-new-launch-001'); await waitFor(() => h.handles.length === 2);
    for (const line of reportLines) old.log.append('stderr', line);
    const status = await h.controller.status();
    assert.equal(status.runnerReport.modelMiB, null);
    assert.equal(status.logs.some((entry) => entry.seq > status.apuObservation.launchSeq && entry.line.includes('buffer size')), false);
    h.ready(); await waitFor(() => h.controller.state.deployment.phase === 'error');
    assert.match(h.controller.state.deployment.error, /gpu_offload_unproven/);
    assert.equal(h.handles[1].killed, true);
});
test('T15: after APU commit, lost GPU refuses new Runs and never creates CPU deployment', async (t) => {
    const h = harness(t); await h.controller.overview();
    h.setSnapshot({ ...snapshot(), gpu: { available: false, state: 'absent', reason: 'render node gone' } });
    await assert.rejects(run(h), /restart local-llm/);
    assert.equal(h.downloads, 0); assert.equal(h.handles.length, 0); assert.equal(h.controller.state.deployment, null);
});
test('T15: shared/dedicated RADV mismatch is refused in both directions', () => {
    const snap = snapshot(); assert.match(verdict({ snap, profile: 'vulkan' }).reason, /restart local-llm/);
    snap.gpu.memoryModel = 'dedicated'; assert.match(verdict({ snap }).reason, /restart local-llm/);
});
for (const phase of ['paused', 'idle', 'error']) test(`T15: restart retains ${phase} APU record without automatically launching`, async (t) => {
    const state = emptyState(); state.deployment = { id: 'retained', modelId: model.id, runnerId: runner.id, profile: 'apu', phase, params: { ctxSize: 4096, parallel: 1 } };
    const h = harness(t, { initialState: state }); await h.controller.status(); await h.controller.overview();
    assert.equal(h.controller.state.deployment.phase, phase); assert.equal(h.handles.length, 0); assert.equal(h.downloads, 0);
});
test('L11: actual load marker and current identities remain sequenced through Stop to idle, without loading phase', async (t) => {
    const h = harness(t, { ready: false, lines: [...reportLines, 'loading model tensors, this can take a while...'] });
    await run(h); await waitFor(() => h.handles.length === 1);
    const starting = await h.controller.status({ sinceSeq: h.controller.state.deployment.logSeqStart });
    assert.equal(starting.phase, 'starting'); assert.equal(starting.apuObservation.running, true);
    const load = starting.logs.find((entry) => entry.line.includes('loading model tensors'));
    assert.ok(load.seq > starting.apuObservation.launchSeq);
    await h.controller.stop(); const stopped = await h.controller.status({ sinceSeq: starting.deployment.logSeqStart });
    assert.equal(stopped.phase, 'idle'); assert.equal(stopped.apuObservation.running, false);
    assert.ok(stopped.logs.some((entry) => entry.line === 'phase idle'));
    assert.equal(stopped.logs.some((entry) => entry.line === 'phase ready'), false);
    for (let i = 1; i < stopped.logs.length; i += 1) assert.equal(stopped.logs[i].seq, stopped.logs[i - 1].seq + 1);
});
test('Stop of a ready APU runner keeps its deployment identity and ends its group', async (t) => {
    const h = harness(t); await run(h); await waitFor(() => h.controller.state.deployment.phase === 'ready');
    const id = h.controller.state.deployment.id;
    await h.controller.stop(); const status = await h.controller.status();
    assert.equal(status.phase, 'idle'); assert.equal(status.deployment.id, id);
    assert.equal(status.deployment.profile, 'apu'); assert.equal(status.apuObservation.running, false);
    assert.equal(h.handles.length, 1);
});

const activity = async (h) => (await h.controller.status()).apuActivity;
function deferred() {
    let resolve; let reject;
    const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
    return { promise, resolve, reject };
}
async function refuse(h, requestId = 'apu-refusal-0001') {
    const snap = snapshot(); snap.memory.availableBytes = 512 * MIB; h.setSnapshot(snap);
    let caught;
    try { await run(h, {}, requestId); } catch (error) { caught = error; }
    assert.equal(caught.code, 'admission_insufficient_now');
    return caught.details.apuAttempt;
}
test('APU activity covers no-deployment initial refusal and distinct refused retries', async (t) => {
    const h = harness(t);
    const before = await activity(h);
    assert.equal(before.schema, 1); assert.equal(before.valid, true);
    assert.equal(before.modelJobsStarted, 0); assert.equal(before.activeModelJobs, 0);
    const attempt = await refuse(h); const retry = await refuse(h);
    assert.equal(attempt.boundary, 'initial-admission'); assert.equal(attempt.requestId, 'apu-refusal-0001');
    assert.equal(attempt.modelId, model.id); assert.equal(attempt.runnerId, runner.id);
    assert.deepEqual(attempt.params, { ctxSize: 4096, parallel: 1 });
    assert.equal(attempt.controllerInstanceId, before.controllerInstanceId);
    assert.notEqual(attempt.attemptId, retry.attemptId);
    for (const sample of [attempt.before, attempt.after, retry.after, await activity(h)]) {
        assert.equal(sample.valid, true); assert.equal(sample.modelJobsStarted, 0);
        assert.equal(sample.runnerInstallsStarted, 0); assert.equal(sample.modelLookupsStarted, 0);
        assert.equal(sample.activeModelJobs, 0); assert.equal(sample.activeRunnerInstalls, 0); assert.equal(sample.activeModelLookups, 0);
        assert.equal(sample.runnerRunning, false);
    }
    assert.equal(h.controller.state.deployment, null);
    assert.equal(h.controller.state.requests['apu-refusal-0001'], undefined);
});
test('APU attempts do not attach to unknown IDs, unsupported runners or invalid parameters', async (t) => {
    const h = harness(t);
    for (const input of [{ modelId: 'missing-model' }, { runnerId: 'missing-runner' }, { runnerId: 'llama.cpp' },
        { params: { nGpuLayers: 0 } }, { requestId: '?' }]) {
        await assert.rejects(h.controller.run({ modelId: model.id, runnerId: runner.id, requestId: 'apu-invalid-0001', ...input }),
            (error) => error.details?.apuAttempt === undefined);
    }
});
test('APU accepted replay does not create an acquisition entry or refused attempt', async (t) => {
    const h = harness(t); await run(h); await waitFor(() => h.controller.state.deployment.phase === 'ready');
    const before = await activity(h); const result = await run(h);
    assert.equal(result.duplicate, true); assert.equal(result.apuAttempt, undefined);
    const after = await activity(h); assert.equal(before.modelJobsStarted, 1); assert.equal(after.modelJobsStarted, 1);
    assert.equal(after.activeModelJobs, 0); assert.equal(after.runnerRunning, true);
});
test('APU activity counts an active model job and a job completing between status polls', async (t) => {
    const gate = deferred(); t.after(() => gate.resolve({ status: 'complete', path: '/tmp/synthetic-gguf', bytesTransferred: 0 }));
    const h = harness(t, { downloadImpl: () => gate.promise });
    await run(h); await waitFor(() => h.downloads === 1);
    const busy = await activity(h); assert.equal(busy.modelJobsStarted, 1); assert.equal(busy.activeModelJobs, 1);
    gate.resolve({ status: 'complete', path: '/tmp/synthetic-gguf', bytesTransferred: 0 });
    await waitFor(() => h.controller.state.deployment.phase === 'ready'); await h.controller.stop();
    const snap = await activity(h); assert.equal(snap.modelJobsStarted, 1); assert.equal(snap.activeModelJobs, 0);
    const attempt = await refuse(h); assert.equal(attempt.before.modelJobsStarted, 1);
    assert.equal(attempt.after.modelJobsStarted, 1); assert.equal(attempt.after.runnerRunning, false);
});
test('APU synchronous acquisition error still records the entry and finalizes the active count', async (t) => {
    const h = harness(t, { downloadImpl: () => { throw new Error('synthetic download error'); } });
    await run(h); await waitFor(() => h.controller.state.deployment.phase === 'error');
    const sample = await activity(h); assert.equal(sample.modelJobsStarted, 1); assert.equal(sample.activeModelJobs, 0);
    assert.equal(sample.runnerRunning, false);
});
test('APU activity counts GGUF/header and metadata-only lookups, including overlap with refusal', async (t) => {
    const gate = deferred(); const calls = [];
    t.after(() => gate.resolve({ header: true }));
    const h = harness(t, { lookup: (args) => { calls.push(args); return args.format === 'gguf' ? gate.promise : { metadata: true }; } });
    const before = await activity(h); const pending = h.controller.lookupModel({ repo: 'test/model', format: 'gguf' });
    const busy = await activity(h); assert.equal(busy.activeModelLookups, 1); assert.equal(busy.modelLookupsStarted, 1);
    const attempt = await refuse(h); assert.equal(attempt.before.activeModelLookups, 1); assert.equal(attempt.after.activeModelLookups, 1);
    await h.controller.lookupModel({ repo: 'test/model', format: 'hf' });
    gate.resolve({ header: true }); await pending;
    const after = await activity(h); assert.equal(after.modelLookupsStarted, 2); assert.equal(after.activeModelLookups, 0);
    assert.equal(before.modelLookupsStarted, 0); assert.equal(calls.length, 2);
});
test('APU lookup entries between polls and immediate lookup errors cannot disappear', async (t) => {
    const h = harness(t, { lookup: (args) => { if (args.fail) throw new Error('synthetic lookup failure'); return { metadata: true }; } });
    const before = await activity(h);
    await h.controller.lookupModel({ format: 'hf' });
    await assert.rejects(h.controller.lookupModel({ format: 'gguf', fail: true }), /synthetic lookup failure/);
    const after = await activity(h); assert.equal(before.modelLookupsStarted, 0); assert.equal(after.modelLookupsStarted, 2);
    assert.equal(after.activeModelLookups, 0);
});
test('APU activity never certifies counter overflow or bad clocks and restart changes instance identity', async (t) => {
    const h = harness(t, { lookup: () => ({}), activityCounterLimit: 1 });
    const first = await activity(h); await h.controller.lookupModel({}); await h.controller.lookupModel({});
    const overflow = await activity(h); assert.equal(overflow.valid, false);
    assert.equal(overflow.modelLookupsStarted, 1); assert.equal(overflow.activeModelLookups, 0);
    for (const clock of [() => NaN, () => -1, () => Infinity, () => { throw new Error('clock missing'); }]) {
        const bad = harness(t, { monotonicNow: clock }); const sample = await activity(bad);
        assert.equal(sample.valid, false); assert.equal(sample.sampledMonotonicMs, null);
    }
    const second = await activity(harness(t)); assert.notEqual(second.controllerInstanceId, first.controllerInstanceId);
});
test('APU floor stop reports current launch, typed cause, log sequence and monotonic settlement', async (t) => {
    const h = harness(t); await run(h); await waitFor(() => h.controller.state.deployment.phase === 'ready');
    h.time.monotonic = 1000; h.memory.availableBytes = 0;
    await waitFor(() => h.controller.state.deployment.phase === 'error');
    const status = await h.controller.status(); const stop = status.apuObservation.guardStop;
    assert.equal(status.apuObservation.controllerInstanceId, status.apuActivity.controllerInstanceId);
    assert.equal(stop.kind, 'host-memory-floor'); assert.equal(stop.availableBytes, 0);
    assert.equal(stop.floorBytes, status.memoryGuard.floorBytes);
    assert.equal(stop.killRequestedMonotonicMs - status.apuObservation.startedMonotonicMs, 1000);
    assert.equal(stop.settlement, 'resolved'); assert.equal(stop.settledMonotonicMs, 1000); assert.equal(stop.valid, true);
    assert.ok(stop.seq > status.apuObservation.launchSeq);
    assert.ok(status.logs.some((entry) => entry.seq === stop.seq && /below the floor/.test(entry.line)));
});
test('APU telemetry-stale stop never becomes a typed floor stop', async (t) => {
    const h = harness(t); await run(h); await waitFor(() => h.controller.state.deployment.phase === 'ready');
    h.time.monotonic = 2500; await waitFor(() => h.controller.state.deployment.phase === 'error');
    assert.equal((await h.controller.status()).apuObservation.guardStop, null);
});

test('APU initial-refusal before/after counters detect a lookup completed inside the attempt', async (t) => {
    let intervening = false; let h;
    h = harness(t, { lookup: () => ({}), onSnapshot: async () => {
        if (intervening) { intervening = false; await h.controller.lookupModel({ format: 'gguf' }); }
    } });
    await activity(h); intervening = true;
    const attempt = await refuse(h);
    assert.equal(attempt.before.modelLookupsStarted, 0); assert.equal(attempt.after.modelLookupsStarted, 1);
    assert.equal(attempt.after.activeModelLookups, 0);
});
test('APU activity counts runner install acquisition and immediate errors through the actual entrypoint', async (t) => {
    const gate = deferred(); t.after(() => gate.resolve());
    let fail = false;
    const entry = { id: runner.id, version: 'b11159', totalBytes: 1, licence: { requiresAcceptance: false } };
    const installer = { lock: { origin: { [runner.id]: 'image' } }, installable: (id) => id === runner.id,
        entryFor: () => entry, pathsFor: () => ({ runDir: '/tmp/synthetic-runner' }),
        describe: async () => ({ installed: false, runnable: false }),
        fetchAll: () => { if (fail) throw new Error('synthetic install failure'); return gate.promise; },
        ensureRunnable: async () => ({ seconds: 0, bytes: 1 }), pruneOtherVersions: async () => {} };
    const h = harness(t, { installer });
    const accepted = await h.controller.installRunner({ runnerId: runner.id }); assert.equal(accepted.accepted, true);
    const during = await activity(h); assert.equal(during.runnerInstallsStarted, 1); assert.equal(during.activeRunnerInstalls, 1);
    gate.resolve(); await waitFor(() => h.controller.state.runnerInstalls[runner.id].phase === 'installed');
    assert.equal((await activity(h)).activeRunnerInstalls, 0);
    fail = true; await h.controller.installRunner({ runnerId: runner.id });
    await waitFor(() => h.controller.state.runnerInstalls[runner.id].phase === 'error');
    const after = await activity(h); assert.equal(after.runnerInstallsStarted, 2); assert.equal(after.activeRunnerInstalls, 0);
});
test('APU attempt ignores stale deployment download bytes and carries only bounded normalized configuration', async (t) => {
    const state = emptyState(); state.deployment = { id: 'old-deployment', profile: 'apu', phase: 'idle', download: { transferred: 123456 } };
    const h = harness(t, { initialState: state }); const attempt = await refuse(h);
    assert.equal(attempt.after.modelJobsStarted, 0);
    assert.deepEqual(Object.keys(attempt).sort(), ['schema', 'attemptId', 'controllerInstanceId', 'requestId', 'modelId', 'runnerId', 'params', 'boundary', 'before', 'after'].sort());
    assert.doesNotMatch(JSON.stringify(attempt), /123456|private-value|apiKey|argv|env/);
});
for (const elapsed of [1000, 1000.001]) test(`APU guard records elapsed ${elapsed} ms without pretending a late kill meets the deadline`, async (t) => {
    const h = harness(t); await run(h); await waitFor(() => h.controller.state.deployment.phase === 'ready');
    h.time.monotonic = elapsed; h.memory.availableBytes = 0;
    await waitFor(() => h.controller.state.deployment.phase === 'error');
    const launch = (await h.controller.status()).apuObservation;
    const observed = launch.guardStop.killRequestedMonotonicMs - launch.startedMonotonicMs;
    assert.equal(observed, elapsed); assert.equal(observed >= 0 && observed <= 1000, elapsed === 1000);
});
test('APU floor receipt remains pending until kill settles and cannot use a later launch callback', async (t) => {
    const gate = deferred(); t.after(() => gate.resolve());
    const h = harness(t, { killImpl: () => gate.promise });
    await run(h); await waitFor(() => h.controller.state.deployment.phase === 'ready');
    h.memory.availableBytes = 0; await waitFor(() => h.handles[0].killed);
    const pending = (await h.controller.status()).apuObservation;
    assert.equal(pending.guardStop.settlement, 'pending'); assert.equal(pending.guardStop.settledMonotonicMs, null);
    await h.controller.stop(); h.memory.availableBytes = 24 * GIB;
    await run(h, {}, 'apu-after-pending-001'); await waitFor(() => h.controller.state.deployment.phase === 'ready');
    const newLaunch = (await h.controller.status()).apuObservation;
    assert.notEqual(newLaunch.deploymentId, pending.deploymentId); assert.equal(newLaunch.guardStop, null);
    gate.resolve(); await new Promise((resolve) => setTimeout(resolve, 10));
    const after = (await h.controller.status()).apuObservation;
    assert.equal(after.deploymentId, newLaunch.deploymentId); assert.equal(after.guardStop, null);
});
test('APU rejected floor kill records rejected settlement without manufacturing a resolved proof', async (t) => {
    const h = harness(t, { killImpl: async (handle) => { await handle.stop(); throw new Error('synthetic kill failure'); } });
    await run(h); await waitFor(() => h.controller.state.deployment.phase === 'ready');
    h.memory.availableBytes = 0; await waitFor(() => h.controller.state.deployment.phase === 'error');
    const stop = (await h.controller.status()).apuObservation.guardStop;
    assert.equal(stop.settlement, 'rejected'); assert.equal(stop.valid, false);
});
test('APU bad kill timestamp invalidates only evidence and still invokes the existing termination', async (t) => {
    let badKillTime = false;
    const h = harness(t, { monotonicNow: () => badKillTime ? -1 : 0 });
    await run(h); await waitFor(() => h.controller.state.deployment.phase === 'ready');
    h.memory.availableBytes = 0;
    // The floor branch logs before reading its evidence clock; use the real log sink as the observation boundary.
    const append = fs.appendFileSync;
    t.mock.method(fs, 'appendFileSync', (file, text, ...args) => {
        if (String(text).includes('below the floor')) badKillTime = true;
        return append(file, text, ...args);
    });
    await waitFor(() => h.controller.state.deployment.phase === 'error');
    const stop = (await h.controller.status()).apuObservation.guardStop;
    assert.equal(stop.killRequestedMonotonicMs, null); assert.equal(stop.valid, false); assert.equal(h.handles[0].killed, true);
});
test('APU activity and attempt fields never change a CPU status or refusal schema', async (t) => {
    const h = harness(t); h.setSnapshot({ ...snapshot(), gpu: { available: false, state: 'absent', reason: 'absent' } });
    const status = await h.controller.status(); assert.equal(status.profile, 'cpu');
    assert.equal('apuActivity' in status, false); assert.equal('apuObservation' in status, false);
    const snap = snapshot(); snap.gpu = { available: false, state: 'absent', reason: 'absent' }; snap.memory.availableBytes = 0;
    h.setSnapshot(snap); await assert.rejects(h.controller.run({ modelId: model.id, runnerId: 'llama.cpp', requestId: 'cpu-refuse-0001' }),
        (error) => error.details?.apuAttempt === undefined);
});
