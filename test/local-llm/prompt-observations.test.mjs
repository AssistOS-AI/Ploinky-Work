import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { createPromptObservations } from '../../lib/local-llm/promptObservations.mjs';
import { createObservationClock, createStartupObservations } from '../../lib/local-llm/controller/startupObservations.mjs';
import { createController } from '../../lib/local-llm/controller/deployments.mjs';
import { loadSeedCatalog, validateModel } from '../../lib/local-llm/controller/catalog.mjs';
import { SEED_CATALOG_FILE } from './fixtures/seedCatalog.mjs';
import { VULKAN_KEYS } from './fixtures/sharedFixtures.mjs';

function observationHarness() {
    let now = 0;
    let current = { deploymentId: 'deployment', runnerIdentity: { runnerStartId: 'runner', pid: 42 },
        target: { modelId: 'model', runnerId: 'runner', baseUrl: 'http://127.0.0.1:1', apiKey: 'private' } };
    const events = createObservationClock({ clock: () => now });
    const observations = createPromptObservations({ events, current: () => current });
    return { observations, events, tick: (ms) => { now += ms; }, replace: () => { current = null; } };
}
const credentials = (begun) => ({ observationId: begun.observation.id, lease: begun.lease });

test('leases bind events to the current runner and epoch, public records contain no private material', () => {
    const h = observationHarness(); const began = h.observations.begin({ observationId: 'prompt_01' });
    assert.throws(() => h.observations.begin({ observationId: 'prompt_01' }), { code: 'observation_duplicate' });
    assert.throws(() => h.observations.record({ observationId: 'prompt_01', lease: 'wrong', event: 'first-token' }), { code: 'observation_stale' });
    const recorded = h.observations.record({ ...credentials(began), event: 'first-token' });
    assert.equal(recorded.recorded, true); assert.equal(recorded.observation.firstTokenPresent, true);
    assert.doesNotMatch(JSON.stringify(h.observations.snapshot()), /private|lease|baseUrl|apiKey/);
    h.replace(); assert.throws(() => h.observations.record({ ...credentials(began), event: 'first-token' }), { code: 'observation_stale' });
    assert.throws(() => observationHarness().observations.finish({ ...credentials(began), outcome: 'success' }), { code: 'observation_stale' });
});

test('active and total capacity refuse without eviction, TTL expiry is incomplete and terminal TTL prunes', () => {
    const h = observationHarness();
    for (let i = 0; i < 16; i++) h.observations.begin({ observationId: `prompt_${i.toString().padStart(2, '0')}` });
    assert.throws(() => h.observations.begin({ observationId: 'overflow_01' }), { code: 'observation_capacity' });
    h.tick(300000); assert.ok(h.observations.snapshot().every((entry) => entry.outcome === 'incomplete'));
    for (let i = 16; i < 64; i++) { const begun = h.observations.begin({ observationId: `prompt_${i}` }); h.observations.finish({ ...credentials(begun), outcome: 'success' }); }
    assert.throws(() => h.observations.begin({ observationId: 'overflow_02' }), { code: 'observation_capacity' });
    h.tick(600000); assert.equal(h.observations.snapshot().length, 0);
});

test('Stop orders first token and failure while completed prompts remain success in the race', () => {
    for (const outcome of ['success', 'failed']) {
        const h = observationHarness(); const begun = h.observations.begin({ observationId: 'prompt_01' });
        h.observations.record({ ...credentials(begun), event: 'first-token' });
        h.observations.stop({ deploymentId: 'deployment', runnerIdentity: { runnerStartId: 'runner' }, sequence: h.events.next() }); h.replace();
        const { observation } = h.observations.finish({ ...credentials(begun), outcome, errorCode: 'runner_unreachable' });
        assert.ok(observation.firstTokenSequence < observation.stopSequence && observation.stopSequence < observation.terminalSequence);
        assert.equal(observation.outcome, outcome === 'success' ? 'success' : 'interrupted-after-stop');
        assert.equal(observation.errorCode, outcome === 'success' ? null : 'runner_unreachable');
    }
});

test('Stop notification is lease-bound, idempotent after idle, and exact cancellation proofs are required', () => {
    const h = observationHarness(); const begun = h.observations.begin({ observationId: 'prompt_01' }); const args = credentials(begun);
    const other = h.observations.begin({ observationId: 'prompt_02' });
    for (const invalid of [{}, { ...args, lease: 'wrong' }, { ...args, observationId: other.observation.id }]) {
        assert.throws(() => h.observations.observeStop(invalid), { code: 'observation_stale' });
    }
    assert.equal(h.observations.observeStop(args).observation.stopNotificationSequence, null);
    const stop = { deploymentId: 'deployment', runnerIdentity: { runnerStartId: 'runner' }, sequence: h.events.next() };
    h.observations.stop(stop); h.observations.stop({ ...stop, sequence: h.events.next() }); h.replace();
    const observed = h.observations.observeStop(args).observation;
    assert.equal(observed.stopSequence, stop.sequence);
    assert.deepEqual(h.observations.observeStop(args).observation, observed);
    const cancellation = { mechanism: 'controller-stop-abort-v1', stopSequence: observed.stopSequence, notificationSequence: observed.stopNotificationSequence };
    for (const change of [{ cancellation: undefined }, { cancellation: { ...cancellation, extra: 1 } },
        { cancellation: { ...cancellation, notificationSequence: 1.1 } }, { cancellation: { ...cancellation, stopSequence: Number.MAX_SAFE_INTEGER + 1 } },
        { cancellation: { ...cancellation, notificationSequence: cancellation.notificationSequence + 1 } },
        { outcome: 'success' }, { errorCode: 'prompt_transport_failed' }]) {
        assert.throws(() => h.observations.finish({ ...args, outcome: 'failed', errorCode: 'prompt_cancelled_by_stop', cancellation, ...change }), { code: 'invalid_observation' });
    }
    const finished = h.observations.finish({ ...args, outcome: 'failed', errorCode: 'prompt_cancelled_by_stop', cancellation });
    assert.deepEqual(finished.observation.cancellation, cancellation);
    assert.ok(observed.stopSequence < observed.stopNotificationSequence && observed.stopNotificationSequence < finished.observation.terminalSequence);
    assert.throws(() => h.observations.observeStop(args), { code: 'observation_stale' });
    h.tick(300000); assert.throws(() => h.observations.observeStop(credentials(other)), { code: 'observation_stale' });
});
const frame = (value) => `data: ${JSON.stringify(value)}\n\n`;
const delta = (value) => frame({ choices: [{ delta: value }] });

async function until(predicate) {
    const deadline = performance.now() + 2000;
    while (performance.now() < deadline) { if (await predicate()) return; await new Promise((resolve) => setTimeout(resolve, 2)); }
    assert.fail('observation not reached');
}

function controllerHarness(t, { profile = 'cpu', health = 503, holdModels = false, drop = () => true, files = null } = {}) {
    const dataDir = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), 'llm-observe-'));
    const weight = path.join(dataDir, 'weight.gguf'); fs.writeFileSync(weight, 'offline');
    const memory = { totalBytes: 8 * 2 ** 30, availableBytes: 7 * 2 ** 30 }; const handles = []; let modelRequests = 0; let downloads = 0; let drops = 0;
    const seed = loadSeedCatalog(SEED_CATALOG_FILE).filter((model) => model.id === 'qwen2.5-0.5b-instruct-q4_k_m').map((model) => validateModel({ ...model, recommended: { venus: { 'llama.cpp-vulkan': { ctxSize: 2048, parallel: 1, loadMode: 'none' } } } }, { seed: true }));
    if (files) {
        const original = seed[0].sources.gguf;
        const shards = [1, 2, 3].map((index) => ({ file: `fixture-0000${index}-of-00003.gguf`, size: 100000000, sha256: original.sha256 }));
        seed[0] = validateModel({ ...seed[0], sources: { gguf: { ...original, file: shards[0].file, size: 300000000, sha256: undefined, shards } } }, { seed: true });
    }
    const controller = createController({ dataDir, seedCatalog: seed, sharedModelsRoot: null, imageContract: { ...VULKAN_KEYS },
        env: { PATH: '/usr/bin' }, fileExists: () => true, installer: { installable: () => false }, detectRunner: () => ({ installed: true }),
        snapshot: async () => ({ memory, disk: { freeBytes: 1e12 }, gpu: profile === 'cpu' ? { available: false, state: 'absent' }
            : { available: true, backend: 'vulkan', driver: 'venus', memoryModel: 'venus', name: 'fixture', totalBytes: 8 * 2 ** 30, freeBytes: 8 * 2 ** 30, icdFile: VULKAN_KEYS.vulkan_icd_venus } }),
        readMemory: () => memory, readPressure: () => ({ some: { avg10: 0 }, full: { avg10: 0 } }), readBudget: () => ({ state: 'absent' }),
        inspect: async () => ({ state: 'complete', bytes: 491400032, total: 491400032 }),
        download: async () => { const target = files ? files(weight, dataDir)[downloads % 3] : weight; downloads++; return { status: 'complete', path: target, bytesTransferred: 0 }; },
        dropCache: (file) => { drops++; return drop(file); }, pollMs: 2, stopGraceMs: 20,
        startRunner() { let resolve; const handle = { pid: 9000 + handles.length, running: true, exited: new Promise((done) => { resolve = done; }),
            async stop() { handle.running = false; resolve({ code: 0, signal: 'SIGTERM' }); }, async kill() { return handle.stop(); } }; handles.push(handle); return handle; },
        fetchImpl: async (url, { signal }) => {
            if (url.endsWith('/health')) { const status = typeof health === 'function' ? health() : health; if (status === 'refused') throw new Error('connection refused'); return { ok: status === 200, status }; }
            modelRequests++;
            if (holdModels) return new Promise((resolve, reject) => { signal.addEventListener('abort', () => reject(new Error('stopped')), { once: true }); });
            return { ok: true, status: 200 };
        },
    });
    t.after(async () => { await controller.drain(); fs.rmSync(dataDir, { force: true, recursive: true }); });
    return { controller, handles, count: () => ({ downloads, drops, modelRequests }),
        run: (requestId, params = { ctxSize: 2048, parallel: 1, loadMode: 'none' }) => controller.run({ requestId, modelId: seed[0].id, runnerId: profile === 'cpu' ? 'llama.cpp' : 'llama.cpp-vulkan', params }) };
}

for (const profile of ['cpu', 'venus']) test(`${profile}: real llama adapter binds 503 to Stop and ignores readiness/connection-only cases`, async (t) => {
    for (const options of [{ health: 503 }, { health: 200, holdModels: true }, { health: 'refused' }]) {
        const h = controllerHarness(t, { profile, ...options }); await h.run(`request_${options.health}`);
        await until(async () => { const status = await h.controller.status(); return options.health === 'refused' ? h.handles.length : status.startupObservation?.latestHealthObservation; });
        const receipt = (await h.controller.stop()).stopObservation;
        assert.equal(receipt.phaseAtStop, 'starting'); assert.equal(receipt.runnerRunning, true); assert.equal(receipt.runnerIdentity.pid, h.handles[0].pid);
        assert.equal(receipt.startupObservation.loadingObserved, options.health === 503);
        assert.equal(receipt.firstAcceptedHealthSequence !== null, options.health === 200);
        if (options.health === 200) assert.ok(h.count().modelRequests > 0, 'accepted health recorded before pending models');
        assert.equal((await h.controller.status()).phase, 'idle'); assert.equal(h.handles[0].running, false);
        assert.deepEqual((await h.controller.status()).stopObservation, receipt);
    }
});

test('no spawned runner cannot certify loading; stale callbacks cannot rewrite Stop receipts', () => {
    assert.equal(createStartupObservations(createObservationClock()).stop(null, null).runnerIdentity, null);
    const events = createObservationClock(); const startup = createStartupObservations(events); startup.reset('deployment');
    const process = { running: true, pid: 1 }; assert.equal(startup.observe('deployment', process, { status: 503, accepted: false }), false);
    const empty = startup.stop({ id: 'deployment', phase: 'starting' }, null); assert.equal(empty.runnerRunning, false); assert.equal(empty.runnerIdentity, null);
    startup.reset('next'); startup.attach('next', process); startup.observe('next', process, { status: 503, accepted: false });
    const receipt = startup.stop({ id: 'next', phase: 'starting' }, process);
    assert.equal(startup.observe('next', process, { status: 200, accepted: true }), false); assert.equal(receipt.firstAcceptedHealthSequence, null);
    startup.reset('replacement'); assert.equal(startup.observe('next', process, { status: 200, accepted: true }), false);
});

for (const profile of ['cpu', 'venus']) test(`${profile}: a received 503 followed by accepted health is too late for loading cancellation`, async (t) => {
    let responses = 0;
    const h = controllerHarness(t, { profile, health: () => ++responses === 1 ? 503 : 200, holdModels: true });
    await h.run('loading_ready');
    await until(async () => (await h.controller.status()).startupObservation?.firstAcceptedHealthSequence !== null);
    const receipt = (await h.controller.stop()).stopObservation;
    assert.equal(receipt.startupObservation.loadingObserved, true); assert.equal(receipt.phaseAtStop, 'starting');
    assert.equal(receipt.latestHealthObservation.accepted, true); assert.ok(receipt.firstAcceptedHealthSequence < receipt.sequence);
});

test('Venus preview and Run use explicit empty parameters instead of saved values, and ignore invalid saved values', async (t) => {
    const h = controllerHarness(t, { profile: 'venus', health: 200 });
    const modelId = 'qwen2.5-0.5b-instruct-q4_k_m'; const runnerId = 'llama.cpp-vulkan';
    await h.run('params_saved', { ctxSize: 4096, parallel: 1, loadMode: 'none' }); await until(async () => (await h.controller.status()).phase === 'ready'); await h.controller.stop();
    assert.equal((await h.controller.overview({ preview: { modelId, runnerId } })).preview.params.ctxSize, 4096);
    const preview = (await h.controller.overview({ preview: { modelId, runnerId, params: {} } })).preview;
    assert.equal(preview.params.ctxSize, 2048); assert.equal(preview.measurementStatus.state, 'unverified');
    assert.equal((await h.run('params_empty', {})).deployment.params.ctxSize, 2048); await h.controller.stop();
    const key = Object.keys(h.controller.state.params)[0]; h.controller.state.params[key] = { arbitrary: true };
    assert.equal((await h.controller.overview({ preview: { modelId, runnerId } })).preview.params.ctxSize, 2048);
});
