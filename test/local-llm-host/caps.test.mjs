// The hardware caps of a pworker home (localLlm.caps, lib/local-llm-host/caps.mjs): parsing, the limits hooks on their own, and the
// host composition end to end. The end-to-end cases run the real llama.cpp runner table the host builds, through the real
// controller, on a fake machine (a hardware snapshot, no GPU, no process): a model is previewed and the admission verdict is read.
// Every capped case has a negative control on the same fake machine and the same models with no caps, which must be admitted, so a
// refusal is known to come from the cap and not from the size of the fake machine. Nothing here signals a process: the reap test
// uses a fake /proc and a recording kill.
import '../home.mjs';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';

import { PassThrough } from 'node:stream';

import { createPworkerClient } from '../../lib/client.mjs';
import { deepFreeze, normalizeWith } from '../../lib/local-llm/runners/params.mjs';
import { loadSeedCatalog } from '../../lib/local-llm/controller/catalog.mjs';
import { LocalLlmError, defineLimits, IDENTITY_LIMITS } from '../../lib/local-llm/index.mjs';
import { capsDigest, defineCapLimits, parseCaps } from '../../lib/local-llm-host/caps.mjs';
import { createLocalLlmGate } from '../../lib/local-llm-host/gate.mjs';
import { admit } from '../../lib/local-llm/controller/admission.mjs';
import { cpuDefaultThreads, defaultThreads, performanceCoreCount, physicalCoreCount } from '../../lib/local-llm/controller/hardware.mjs';
import { darwinPlatform } from '../../lib/local-llm/darwinPlatform.mjs';
import {
    LLAMA_SERVER_CPU_PARAM_SCHEMA, LLAMA_SERVER_METAL_PARAM_SCHEMA, LLAMA_SERVER_PARAM_SCHEMA, LLAMA_SERVER_UNIFIED_PARAM_SCHEMA,
} from '../../lib/local-llm/runners/llamaServer.mjs';
import { cpuFloorBytes, cpuHostReserveBytes } from '../../lib/local-llm/controller/profiles.mjs';
import { getRunner } from '../local-llm/fixtures/host.mjs';
import { SEED_CATALOG_FILE, createLocalLlmHost } from '../../lib/local-llm-host/host.mjs';
import { LAUNCH_SCHEMA } from '../../lib/local-llm-host/launch.mjs';
import { capsLine, runEngineVerb } from '../../lib/pworker/models-cli.mjs';
import { statusFor } from '../../lib/local-llm-host/routes.mjs';
import { GIB, MODEL, fakeAdapter, freePort, scratch } from './fixtures/engine.mjs';
import { request, startHost, startProxy } from './fixtures/proxy.mjs';
import { recordingKill, removeProc, runnerArgv, writeProc } from './fixtures/proc.mjs';

const amount = (bytes) => (bytes >= GIB ? `${(bytes / GIB).toFixed(2)} GiB` : `${Math.round(bytes / (1024 * 1024))} MiB`);

// ---------------------------------------------------------------- the fake machines

const DISK = { freeBytes: 500 * GIB, totalBytes: 900 * GIB };
const MEMORY = { totalBytes: 64 * GIB, availableBytes: 60 * GIB };
const CPU_MACHINE = { gpu: { available: false, state: 'absent' }, memory: MEMORY, disk: DISK, cpus: 16 };
const DEDICATED_MACHINE = {
    gpu: { available: true, name: 'Test GPU', driverVersion: '580.1', memoryModel: 'dedicated', totalBytes: 24 * GIB, usedBytes: 0, freeBytes: 24 * GIB, processes: [] },
    memory: MEMORY, disk: DISK, cpus: 16,
};
const UNIFIED_MACHINE = {
    gpu: { available: true, name: 'Test APU', driverVersion: '580.1', memoryModel: 'unified', totalBytes: null, usedBytes: null, freeBytes: null, processes: [],
        device: { pciDeviceId: '0x2E1210DE', computeCapability: '12.1', addressingMode: 'ATS' } },
    memory: MEMORY, disk: DISK, cpus: 16,
};
const RUNNER = { cpu: 'llama.cpp-cpu', dedicated: 'llama.cpp', unified: 'llama.cpp' };

/** The catalog's first model with other pinned weights: the size of the weights is what the policies scale with. */
function modelOf(id, weightsGiB) {
    const model = structuredClone(loadSeedCatalog(SEED_CATALOG_FILE)[0]);
    model.id = id;
    model.sources.gguf.size = weightsGiB * GIB;
    model.sources.gguf.file = `${id}.gguf`;
    return model;
}
const SIZES = { tiny: 0.25, small: 2, six: 6, ten: 10, twelve: 12 };
// Models with a known training context per slot (8192), the second recommending four slots on the cpu profile.
const CONTEXT = 8192;
const withContext = (model, extra = {}) => Object.assign(model, { contextLength: CONTEXT }, extra);
const SEED = [
    ...Object.entries(SIZES).map(([id, gib]) => modelOf(id, gib)),
    withContext(modelOf('slots', 2)),
    withContext(modelOf('slots-recommended', 2), { recommended: { cpu: { 'llama.cpp-cpu': { parallel: 4 } } } }),
];

/** A host on a scratch home with the real runner table, a fake machine and no process. `caps` undefined means no caps. */
async function capsHost(t, { caps, profile, machine, killImpl, procDir, home = scratch(t) }) {
    const config = caps === undefined ? {} : { localLlm: { caps } };
    const host = createLocalLlmHost({
        config, home, env: { PATH: '/usr/bin:/bin', HOME: '/home/test', LANG: 'C.UTF-8' }, platformName: 'linux',
        procDir: procDir ?? path.join(home, 'proc'), ownerPort: await freePort(), which: () => '/usr/bin/nvidia-smi', ...(killImpl ? { killImpl } : {}),
        controllerOptions: {
            profile, snapshot: async () => structuredClone(machine), seedCatalog: SEED,
            inspect: async () => ({ state: 'complete', bytes: 1 }), detectRunner: () => ({ installed: true, version: 'v1', reason: null }),
        },
    });
    t.after(() => host.drain().catch(() => {}));
    await host.acquire();
    return host;
}

/** The admission verdict for a model of SIZES on the host's real controller. */
async function previewOf(host, profile, modelId) {
    const controller = await host.controller();
    const { preview } = await controller.overview({ preview: { modelId, runnerId: RUNNER[profile] } });
    return preview.admission;
}

// ---------------------------------------------------------------- V1: parsing

test('parseCaps: no section is no caps, and sizes are bytes, strings with binary units, or fractions', () => {
    assert.equal(parseCaps(undefined), null);
    assert.equal(parseCaps(null), null);
    assert.equal(parseCaps({}), null, 'an empty section caps nothing');
    assert.equal(parseCaps({ mode: 'refuse' }), null);
    assert.equal(parseCaps({ ram: '12GiB' }).ram.bytes, 12884901888);
    assert.equal(parseCaps({ ram: '512MiB' }).ram.bytes, 512 * 1024 * 1024);
    assert.equal(parseCaps({ gpuMemory: '1.5GiB' }).gpuMemory.bytes, 1610612736);
    assert.equal(parseCaps({ ram: 8 * GIB }).ram.bytes, 8 * GIB);
    assert.equal(parseCaps({ ram: 1048576 }).ram.bytes, 1048576, '1 MiB is the smallest size');
    assert.deepEqual({ ...parseCaps({ gpuMemory: 0.5 }).gpuMemory }, { fraction: 0.5 });
    const caps = parseCaps({ ram: '8GiB', gpuMemory: '4GiB', mode: 'refuse' });
    assert.equal(caps.mode, 'refuse');
    assert.ok(Object.isFrozen(caps));
});

test('parseCaps refuses bad values, unknown keys and the keys of later releases, each error naming the key', () => {
    const cases = [
        [{ ram: 0 }, 'ram'], [{ ram: -5 }, 'ram'], [{ ram: 1.5 }, 'ram'], [{ ram: true }, 'ram'], [{ ram: '12' }, 'ram'],
        [{ ram: '0GiB' }, 'ram'], [{ ram: '12 GB' }, 'ram'], [{ gpuMemory: 'lots' }, 'gpuMemory'], [{ gpuMemory: null }, 'gpuMemory'],
        [{ gpuMemory: Number.MAX_SAFE_INTEGER + 2 }, 'gpuMemory'], [{ bogus: 1 }, 'bogus'],
        [{ mode: 'fit' }, 'mode'],
        [{ ram: '8GiB', mode: 'strict' }, 'mode'], [{ gpuMemory: 12 }, 'gpuMemory'], [{ ram: 1048575 }, 'ram'], [{ ram: '512KiB' }, 'ram'],
        [{ threads: 0 }, 'threads'], [{ threads: -2 }, 'threads'], [{ threads: 1.5 }, 'threads'], [{ threads: '4' }, 'threads'], [{ threads: null }, 'threads'],
        [{ maxParallel: 0 }, 'maxParallel'], [{ maxParallel: 2.5 }, 'maxParallel'], [{ maxParallel: true }, 'maxParallel'],
        [{ maxContext: 0 }, 'maxContext'], [{ maxContext: 511 }, 'maxContext'], [{ maxContext: '4096' }, 'maxContext'], [{ maxContext: 4096.5 }, 'maxContext'],
    ];
    for (const [section, key] of cases) {
        assert.throws(() => parseCaps(section), (error) => error instanceof LocalLlmError && error.code === 'invalid_caps'
            && error.details.key === key && error.message.includes(key), JSON.stringify(section));
    }
    assert.throws(() => parseCaps({ mode: 'fit' }), /not supported yet/);
    assert.throws(() => parseCaps({ maxContext: 511 }), /smallest context a runner accepts/);
    assert.throws(() => parseCaps([]), (error) => error.code === 'invalid_caps');
    assert.throws(() => parseCaps('8GiB'), (error) => error.code === 'invalid_caps');
});

test('parseCaps accepts threads, maxContext and maxParallel as whole numbers, alone or with the sizes, and maxContext down to the smallest runner minimum', () => {
    assert.deepEqual({ ...parseCaps({ threads: 2 }) }, { threads: 2, mode: 'refuse' });
    assert.deepEqual({ ...parseCaps({ maxContext: 4096, maxParallel: 1 }) }, { maxContext: 4096, maxParallel: 1, mode: 'refuse' });
    const all = parseCaps({ ram: '8GiB', threads: 4, maxContext: 8192, maxParallel: 2, mode: 'refuse' });
    assert.equal(all.threads, 4);
    assert.equal(all.ram.bytes, 8 * GIB);
    // The smallest ctxSize minimum among the runner schemas pworker offers is 512; a cap at it is the lowest valid one.
    const minimum = Math.min(...[LLAMA_SERVER_PARAM_SCHEMA, LLAMA_SERVER_UNIFIED_PARAM_SCHEMA, LLAMA_SERVER_CPU_PARAM_SCHEMA, LLAMA_SERVER_METAL_PARAM_SCHEMA].map((schema) => schema.properties.ctxSize.minimum));
    assert.equal(parseCaps({ maxContext: minimum }).maxContext, minimum);
    assert.throws(() => parseCaps({ maxContext: minimum - 1 }), { code: 'invalid_caps' });
});

test('a fraction is resolved against the first physical figure seen and then kept', () => {
    const limits = defineCapLimits(parseCaps({ ram: 0.5 }));
    const first = limits.memoryPool({ totalBytes: 32 * GIB, availableBytes: 30 * GIB }, null, { amount });
    assert.equal(first.totalBytes, 16 * GIB);
    assert.equal(first.availableBytes, 16 * GIB, 'available is capped to the cap too');
    const later = limits.memoryPool({ totalBytes: 64 * GIB, availableBytes: 10 * GIB }, null, { amount });
    assert.equal(later.totalBytes, 16 * GIB, 'frozen at the first reading');
    assert.equal(later.availableBytes, 10 * GIB, 'what is available now stays when it is below the cap');
});

// ---------------------------------------------------------------- the hooks on their own

test('defineCapLimits binds the memory pool, the memory view, the GPU share and the overview, and leaves every other hook the identity', () => {
    assert.throws(() => defineCapLimits(null), (error) => error.code === 'invalid_caps');
    const limits = defineCapLimits(parseCaps({ ram: '8GiB', gpuMemory: '4GiB' }));
    for (const hook of ['capCores', 'observeSnapshot', 'decorateSnapshot', 'refuseProfile', 'refuseResourceRequest', 'runnerEnvironment', 'guardSample', 'guardBreach']) {
        assert.equal(limits[hook], IDENTITY_LIMITS[hook], hook);
    }
    const pool = limits.memoryPool({ totalBytes: 64 * GIB, availableBytes: 60 * GIB }, null, { amount });
    assert.deepEqual({ total: pool.totalBytes, available: pool.availableBytes, capped: pool.capped }, { total: 8 * GIB, available: 8 * GIB, capped: true });
    assert.deepEqual({ total: pool.reserveTotalBytes, available: pool.reserveAvailableBytes }, { total: 64 * GIB, available: 60 * GIB }, 'the pool carries the physical figures');
    assert.match(pool.warning, /ram cap/);
    assert.ok(pool.warning.includes('8.00 GiB'));
    // A cap above the machine changes nothing the machine does not already say.
    const roomy = defineCapLimits(parseCaps({ ram: '128GiB' }));
    assert.equal(roomy.memoryPool({ totalBytes: 64 * GIB, availableBytes: 60 * GIB }, null, { amount }).totalBytes, 64 * GIB);

    // dedicated: the share keeps every field of the GPU, including the device total, and caps the total and the free figure.
    const gpu = { ...DEDICATED_MACHINE.gpu, deviceTotalBytes: 24 * GIB, freeBytes: 20 * GIB };
    const share = limits.admit({ stage: 'gpu', selected: 'dedicated', gpu, snapshot: DEDICATED_MACHINE, profile: 'dedicated' });
    assert.deepEqual(share, { share: { gpu: { ...gpu, totalBytes: 4 * GIB, freeBytes: 4 * GIB } } });
    const small = { ...gpu, totalBytes: 3 * GIB, freeBytes: 1 * GIB };
    assert.deepEqual(limits.admit({ stage: 'gpu', selected: 'dedicated', gpu: small, snapshot: DEDICATED_MACHINE }).share.gpu, small, 'a GPU under the cap is as it was');
    assert.equal(limits.admit({ stage: 'gpu', selected: 'unified', gpu, snapshot: UNIFIED_MACHINE }), null, 'unified is capped through the memory view');
    for (const stage of ['before-policy', 'vulkan']) assert.equal(limits.admit({ stage, selected: 'dedicated', gpu }), null, stage);
    assert.equal(defineCapLimits(parseCaps({ ram: '8GiB' })).admit({ stage: 'gpu', selected: 'dedicated', gpu }), null, 'no gpuMemory cap, no share');

    // The memory view: ram on every GPU profile, and on a unified GPU the smaller of ram and gpuMemory.
    const physical = { reserveTotalBytes: 64 * GIB, reserveAvailableBytes: 60 * GIB };
    assert.deepEqual(limits.effective({ snapshot: DEDICATED_MACHINE }).memory, { totalBytes: 8 * GIB, availableBytes: 8 * GIB, ...physical }, 'the physical figures stay beside the capped view');
    assert.deepEqual(limits.effective({ snapshot: UNIFIED_MACHINE }).memory, { totalBytes: 4 * GIB, availableBytes: 4 * GIB, ...physical });
    assert.deepEqual(defineCapLimits(parseCaps({ gpuMemory: '4GiB' })).effective({ snapshot: DEDICATED_MACHINE }).memory, MEMORY, 'gpuMemory leaves a dedicated GPU\'s host memory alone');
    const metal = { ...UNIFIED_MACHINE, gpu: { available: true, backend: 'metal', name: 'Apple M1', memoryModel: 'unified', deviceCapBytes: 12 * GIB } };
    assert.deepEqual(limits.effective({ snapshot: metal }).memory, { totalBytes: 8 * GIB, availableBytes: 8 * GIB, ...physical }, 'a Metal device is not an NVIDIA GPU: only ram applies');
    assert.deepEqual(limits.overview({ snapshot: metal }), { caps: { ram: 8 * GIB }, mode: 'refuse' }, 'and gpuMemory is not published where it does nothing');
    assert.deepEqual(limits.overview({ snapshot: CPU_MACHINE }), { caps: { ram: 8 * GIB }, mode: 'refuse' }, 'nor on a cpu machine');
    assert.equal(limits.admit({ stage: 'gpu', selected: 'dedicated', gpu: metal.gpu, snapshot: metal }), null);

    assert.deepEqual(limits.overview({ snapshot: DEDICATED_MACHINE }), { caps: { ram: 8 * GIB, gpuMemory: 4 * GIB }, mode: 'refuse' });
    const fractional = defineCapLimits(parseCaps({ gpuMemory: 0.25 }));
    assert.deepEqual(fractional.overview({ snapshot: DEDICATED_MACHINE }), { caps: { gpuMemory: 6 * GIB }, mode: 'refuse' });
    assert.deepEqual(fractional.overview({ snapshot: { ...DEDICATED_MACHINE, gpu: { ...DEDICATED_MACHINE.gpu, totalBytes: 48 * GIB } } }).caps, { gpuMemory: 6 * GIB }, 'frozen at the first GPU seen');
    assert.deepEqual(defineCapLimits(parseCaps({ gpuMemory: 0.25 })).overview({ snapshot: UNIFIED_MACHINE }).caps, { gpuMemory: 16 * GIB }, 'on a unified GPU a share of system memory');
    assert.ok(defineLimits({ memoryPool: limits.memoryPool }), 'the hooks are plain functions');
});

test('capThreads and capParams: a count cap lowers a default, refuses an explicit value, leaves null alone, and with no count cap they are the identity', () => {
    const limits = defineCapLimits(parseCaps({ threads: 4, maxContext: 2048, maxParallel: 2 }));
    assert.equal(limits.capThreads(14, { profile: 'cpu' }), 4);
    assert.equal(limits.capThreads(3, { profile: 'metal' }), 3, 'a default under the cap is kept');
    const params = Object.freeze({ ctxSize: 8192, parallel: 4, threads: null, batchSize: 512 });
    assert.deepEqual(limits.capParams({ runnerId: 'llama.cpp', profile: 'dedicated', params, explicit: Object.freeze([]), model: {} }),
        { ctxSize: 2048, parallel: 2, threads: null, batchSize: 512 }, 'defaults and saved values are lowered; threads null is left to capThreads');
    const under = Object.freeze({ ctxSize: 1024, parallel: 1, threads: 3 });
    assert.equal(limits.capParams({ params: under, explicit: Object.freeze(['ctxSize', 'threads']) }), under, 'values under the caps: the same object comes back');
    for (const [key, value, cap] of [['ctxSize', 4096, 2048], ['parallel', 3, 2], ['threads', 8, 4]]) {
        const capKey = { ctxSize: 'maxContext', parallel: 'maxParallel', threads: 'threads' }[key];
        assert.throws(() => limits.capParams({ params: { ctxSize: 1024, parallel: 1, threads: 1, [key]: value }, explicit: Object.freeze([key]) }),
            (error) => error instanceof LocalLlmError && error.code === 'invalid_params' && error.message.includes(`localLlm.caps.${capKey}`)
                && error.message.includes(String(value)) && error.message.includes(String(cap)), key);
    }
    // A context sized for more slots than the cap allows is lowered with them: the training context is per slot.
    assert.equal(limits.capParams({ params: { ctxSize: 8192, parallel: 4 }, explicit: Object.freeze([]), model: { contextLength: 1024 } }).ctxSize, 2048);
    const sized = defineCapLimits(parseCaps({ maxParallel: 1 }));
    assert.equal(sized.capParams({ params: { ctxSize: 8192, parallel: 4 }, explicit: Object.freeze([]), model: { contextLength: 2048 } }).ctxSize, 2048);
    // With only a size cap, the two hooks are the identity.
    const sizes = defineCapLimits(parseCaps({ ram: '8GiB' }));
    assert.equal(sizes.capThreads(14, { profile: 'cpu' }), 14);
    assert.equal(sizes.capParams({ params, explicit: Object.freeze(['ctxSize']) }), params);
    assert.deepEqual(limits.overview({ snapshot: CPU_MACHINE }), { caps: { threads: 4, maxContext: 2048, maxParallel: 2 }, mode: 'refuse' });
    assert.equal(capsLine({ limits: { budget: limits.overview({ snapshot: CPU_MACHINE }) } }), 'caps in force (localLlm.caps): threads 4, maxContext 2048, maxParallel 2; refuse mode');
});

test('capsDigest is stable for equal caps and differs for different ones', () => {
    const digest = capsDigest(parseCaps({ ram: '8GiB' }));
    assert.match(digest, /^[0-9a-f]{12}$/);
    assert.equal(capsDigest(parseCaps({ ram: 8 * GIB })), digest, 'the same bytes, however they were written');
    assert.equal(capsDigest(parseCaps({ ram: '8GiB', mode: 'refuse' })), digest);
    assert.notEqual(capsDigest(parseCaps({ ram: '9GiB' })), digest);
    assert.notEqual(capsDigest(parseCaps({ gpuMemory: '8GiB' })), digest, 'the key counts, not only the number');
    assert.notEqual(capsDigest(parseCaps({ ram: '8GiB', gpuMemory: '8GiB' })), digest);
    assert.notEqual(capsDigest(parseCaps({ ram: '8GiB', threads: 2 })), digest, 'a count cap counts');
    assert.notEqual(capsDigest(parseCaps({ threads: 2 })), capsDigest(parseCaps({ maxParallel: 2 })), 'and so does its key');
    assert.equal(capsDigest(parseCaps({ threads: 2, mode: 'refuse' })), capsDigest(parseCaps({ threads: 2 })));
    assert.equal(capsDigest(null), null);
});

// ---------------------------------------------------------------- the host: a bad config

test('a bad caps section is kept at creation and thrown by controller() after the reap has run; nothing is created', async (t) => {
    const home = scratch(t);
    const procDir = path.join(home, 'proc');
    const sock = path.join(home, 'r', 'abcdef09.sock');
    const launchDir = path.join(home, 'local-llm', 'state', 'launch');
    writeProc(procDir, { pid: 4300, startTime: '5200', argv: runnerArgv(sock) });
    // The launch record of an orphan the reap must end: it is written once acquire() has made the layout.
    const kill = recordingKill({ dies: { SIGTERM: (pid) => removeProc(procDir, -pid) } });
    const config = { localLlm: { caps: { ram: 0 } } };
    fs.mkdirSync(launchDir, { recursive: true });
    fs.writeFileSync(path.join(launchDir, 'abcdef09.json'), `${JSON.stringify({ schema: LAUNCH_SCHEMA, startId: 'abcdef09', socketPath: sock, phase: 'spawned', pid: 4300, startTime: '5200' })}\n`);
    const host = createLocalLlmHost({ config, home, env: {}, platformName: 'linux', procDir, ownerPort: await freePort(), killImpl: kill });
    t.after(() => host.drain().catch(() => {}));
    assert.equal(host.caps, null);
    assert.equal(host.capsRefusal.code, 'invalid_caps', 'creating the host does not throw');
    const acquired = await host.acquire();
    assert.equal(acquired.owned, true);
    assert.deepEqual(kill.calls, [[-4300, 'SIGTERM']], 'the reap ran at acquire, whatever the caps said');
    assert.deepEqual(acquired.reaped.reaped, ['abcdef09']);
    await assert.rejects(host.controller(), (error) => error instanceof LocalLlmError && error.code === 'invalid_caps' && error.details.key === 'ram');
    assert.equal(host.current(), null, 'no controller was created');
    await assert.rejects(host.controller(), { code: 'invalid_caps' }, 'and it is refused again, not remembered as a controller');
});

test('mode fit is refused the same way when the host is used, and the host with no caps has the identity limits', async (t) => {
    const host = await capsHost(t, { caps: { ram: '8GiB', mode: 'fit' }, profile: 'cpu', machine: CPU_MACHINE });
    await assert.rejects(host.controller(), (error) => error.code === 'invalid_caps' && error.details.key === 'mode' && /not supported yet/.test(error.message));
    const plain = await capsHost(t, { profile: 'cpu', machine: CPU_MACHINE });
    assert.equal(plain.caps, null);
    assert.equal(plain.capsRefusal, null);
    const overview = await (await plain.controller()).overview();
    assert.equal(overview.limits?.budget, undefined, 'no caps: the overview has no budget');
});

// ---------------------------------------------------------------- V2 and V5: ram on the cpu profile

test('V2: ram 8GiB on a roomy cpu machine refuses a 12 GiB model as incompatible and says why, and admits a 2 GiB one', async (t) => {
    const host = await capsHost(t, { caps: { ram: '8GiB' }, profile: 'cpu', machine: CPU_MACHINE });
    const big = await previewOf(host, 'cpu', 'twelve');
    assert.equal(big.status, 'incompatible', big.reason);
    assert.ok(big.estimate.ramBytes > 12 * GIB, `the model needs ${big.estimate.ramBytes}`);
    assert.match(big.reason, /8\.00 GiB/, 'the capped figure is in the reason');
    assert.ok(big.warnings.some((line) => /ram cap/.test(line) && line.includes('8.00 GiB')), big.warnings.join(' | '));
    const small = await previewOf(host, 'cpu', 'small');
    assert.equal(small.status, 'ok', small.reason);
    assert.ok(small.estimate.ramBytes < 4 * GIB);
    assert.ok(small.warnings.some((line) => /ram cap/.test(line)), 'the warning accompanies an admitted run too');

    const overview = await (await host.controller()).overview();
    assert.deepEqual(overview.limits.budget, { caps: { ram: 8 * GIB }, mode: 'refuse' });
    assert.equal(capsLine(overview), 'caps in force (localLlm.caps): ram 8 GiB; refuse mode');
});

test('V5: the same cpu machine and models with no caps admit the 12 GiB model, and the overview has no budget', async (t) => {
    const host = await capsHost(t, { profile: 'cpu', machine: CPU_MACHINE });
    const big = await previewOf(host, 'cpu', 'twelve');
    assert.equal(big.status, 'ok', big.reason);
    assert.ok(big.estimate.ramBytes > 12 * GIB);
    assert.ok(!big.warnings.some((line) => /ram cap/.test(line)));
    const overview = await (await host.controller()).overview();
    assert.equal(overview.limits?.budget, undefined);
    assert.equal(capsLine(overview), null);
});

// ---------------------------------------------------------------- V3 and V5: gpuMemory

test('V3: gpuMemory 8GiB on a 24 GiB dedicated GPU refuses a model that needs about 10 GiB and admits one that needs about 6', async (t) => {
    const host = await capsHost(t, { caps: { gpuMemory: '8GiB' }, profile: 'dedicated', machine: DEDICATED_MACHINE });
    const big = await previewOf(host, 'dedicated', 'ten');
    assert.equal(big.status, 'incompatible', big.reason);
    assert.ok(big.estimate.gpuBytes > 10 * GIB && big.estimate.gpuBytes < 12 * GIB, `needs ${big.estimate.gpuBytes}`);
    assert.match(big.reason, /the GPU has 8\.\d+ GiB|the GPU has 8 GiB/, 'the capped figure is the GPU the policy saw');
    const small = await previewOf(host, 'dedicated', 'six');
    assert.equal(small.status, 'ok', small.reason);
    assert.ok(small.estimate.gpuBytes > 6 * GIB && small.estimate.gpuBytes < 8 * GIB, `needs ${small.estimate.gpuBytes}`);
    assert.deepEqual((await (await host.controller()).overview()).limits.budget, { caps: { gpuMemory: 8 * GIB }, mode: 'refuse' });
});

test('V3: gpuMemory 8GiB on a unified GPU refuses the 10 GiB model through the memory view, and a larger cap admits it', async (t) => {
    const host = await capsHost(t, { caps: { gpuMemory: '8GiB' }, profile: 'unified', machine: UNIFIED_MACHINE });
    const big = await previewOf(host, 'unified', 'ten');
    assert.equal(big.status, 'incompatible', big.reason);
    assert.ok(big.estimate.unifiedBytes > 10 * GIB, `needs ${big.estimate.unifiedBytes}`);
    assert.equal(big.estimate.poolBytes, 8 * GIB, 'the policy sized against the capped memory');
    // The unified policy keeps a fixed host reserve (16 GiB) out of the pool, so a cap that leaves room above it admits.
    const roomy = await capsHost(t, { caps: { gpuMemory: '48GiB' }, profile: 'unified', machine: UNIFIED_MACHINE });
    const fits = await previewOf(roomy, 'unified', 'ten');
    assert.equal(fits.status, 'ok', fits.reason);
    assert.equal(fits.estimate.poolBytes, 48 * GIB);
});

test('V3: ram caps the host memory of a dedicated GPU too', async (t) => {
    const host = await capsHost(t, { caps: { ram: '1GiB' }, profile: 'dedicated', machine: DEDICATED_MACHINE });
    const refused = await previewOf(host, 'dedicated', 'six');
    assert.equal(refused.status, 'insufficient-now', refused.reason);
    assert.match(refused.reason, /of RAM; 1\.0+ GiB is available now/, 'the RAM figure is the capped one');
    assert.equal((await previewOf(await capsHost(t, { profile: 'dedicated', machine: DEDICATED_MACHINE }), 'dedicated', 'six')).status, 'ok', 'and the same model fits with no cap');
});

test('V5: the same dedicated and unified machines with no caps admit the 10 GiB model and have no budget', async (t) => {
    for (const [profile, machine] of [['dedicated', DEDICATED_MACHINE], ['unified', UNIFIED_MACHINE]]) {
        const host = await capsHost(t, { profile, machine });
        const big = await previewOf(host, profile, 'ten');
        assert.equal(big.status, 'ok', `${profile}: ${big.reason}`);
        assert.equal((await (await host.controller()).overview()).limits?.budget, undefined, profile);
    }
});

// ---------------------------------------------------------------- V2b and V5: threads, maxContext and maxParallel on the launch

const MAC_MACHINE = {
    gpu: { available: true, backend: 'metal', name: 'Apple M1', memoryModel: 'unified', deviceCapBytes: 12 * GIB },
    memory: { totalBytes: 16 * GIB, availableBytes: 12 * GIB, pressureLevel: 1 }, disk: DISK, cpus: 8, cores: 4,
};
// The captured sysctl of a 16 GiB M1, with the answer `sysctl -n hw.perflevel0.physicalcpu` gives a Mac whose performance cores number 8.
const MAC_EXEC = ((captured) => (file, args, ...rest) => (args?.[0] === '-n' && args[1] === 'hw.perflevel0.physicalcpu' ? '8\n' : captured(file, args, ...rest)))((() => {
    const fixture = (name) => fs.readFileSync(new URL(`./fixtures/darwin/${name}`, import.meta.url), 'utf8');
    const known = new Map(`${fixture('sysctl.txt')}${fixture('swapusage.txt')}`.split('\n').filter(Boolean).map((line) => [line.slice(0, line.indexOf(':')), line]));
    return (file, args) => (file === '/usr/bin/vm_stat' ? fixture('vm_stat.txt') : `${args.filter((key) => known.has(key)).map((key) => known.get(key)).join('\n')}\n`);
})());

/** A fake /proc of 16 physical cores (the sysfs of this machine is not read for cpus that are not there). */
function sixteenCores(home) {
    const procDir = path.join(home, 'proc');
    fs.mkdirSync(path.join(procDir, 'self'), { recursive: true });
    fs.writeFileSync(path.join(procDir, 'self', 'status'), 'Cpus_allowed_list:\t0-15\n');
    fs.writeFileSync(path.join(procDir, 'cpuinfo'), Array.from({ length: 16 }, (_, cpu) => `processor\t: ${cpu}\nphysical id\t: 0\ncore id\t\t: ${cpu}\n`).join('\n'));
    return procDir;
}

/**
 * The default thread count the engine's own functions give on each profile, for the fake machine: what a launch with no caps and
 * no `threads` gets. Nothing here is a hard-coded number.
 */
function defaultThreadsOf(profile, { procDir }) {
    const perf = profile === 'metal' ? darwinPlatform({ exec: MAC_EXEC }).performanceCores() : performanceCoreCount({ procDir });
    if (profile === 'metal' || profile === 'unified') return perf ?? defaultThreads(physicalCoreCount({ procDir }));
    if (profile === 'cpu') return perf ?? cpuDefaultThreads(physicalCoreCount({ procDir }));
    return defaultThreads(physicalCoreCount({ procDir }));
}

/**
 * Runs `small` on a host with the real runner table and the real controller, on a fake machine, with the runner process stubbed
 * out, and returns the llama-server argv it would have been started with (or the error the Run was refused with).
 */
async function launchOf(t, { caps, profile, params, modelId = 'small', home = scratch(t), drain = false }) {
    const procDir = sixteenCores(home);
    const metal = profile === 'metal';
    const machine = { cpu: CPU_MACHINE, dedicated: DEDICATED_MACHINE, unified: UNIFIED_MACHINE, metal: MAC_MACHINE }[profile];
    const started = [];
    const weights = path.join(home, 'weights.gguf');
    fs.writeFileSync(weights, 'weights');
    const controllerOptions = {
        profile, snapshot: async () => structuredClone(machine), seedCatalog: SEED, fileExists: () => true, installer: {
            installable: () => true, lock: null, describe: async () => ({ installed: true, runnable: true, version: 'v1' }),
            ensureRunnable: async () => ({ rebuilt: false }), entryFor: (id) => ({ id }), pathsFor: () => ({ runDir: '/run/llama.cpp/b1' }),
        },
        inspect: async () => ({ state: 'complete', bytes: 1 }), detectRunner: () => ({ installed: true, version: 'v1', reason: null }),
        download: async () => ({ status: 'complete', path: weights, bytesTransferred: 0 }),
        fetchImpl: async () => ({ ok: true, status: 200, json: async () => ({}) }), pollMs: 2, stopGraceMs: 20,
        startRunner(options) {
            started.push(options.args);
            let resolve;
            const handle = { pid: 9000, running: true, exited: new Promise((done) => { resolve = done; }),
                async stop() { handle.running = false; resolve({ code: 0, signal: 'SIGTERM' }); }, async kill() { return handle.stop(); } };
            return handle;
        },
    };
    const config = caps === undefined ? {} : { localLlm: { caps } };
    const host = createLocalLlmHost({
        config, home, env: { PATH: '/usr/bin:/bin', HOME: '/home/test', LANG: 'C.UTF-8' }, platformName: metal ? 'darwin' : 'linux', ...(metal ? { arch: 'arm64', darwinExec: MAC_EXEC } : {}),
        procDir, ownerPort: await freePort(), which: () => '/usr/bin/nvidia-smi', controllerOptions,
    });
    t.after(() => host.drain().catch(() => {}));
    await host.acquire();
    const controller = await host.controller();
    const result = await controller.run({ modelId, runnerId: RUNNER[profile] ?? 'llama.cpp', requestId: `caps-${profile}-${started.length}-${Math.random().toString(16).slice(2, 8)}`, params }).then(() => null, (error) => error);
    if (result) return { error: result, procDir };
    const deadline = Date.now() + 3000;
    while (!started.length && Date.now() < deadline) await new Promise((resolve) => setTimeout(resolve, 5));
    assert.equal(started.length, 1, `the runner was started (${controller.state.deployment?.phase}: ${controller.state.deployment?.error})`);
    // The weights path, the socket and the key are different on every start: the rest of the argv is what the cap can change.
    const masked = started[0].map((arg, index) => (['-m', '--host', '--api-key'].includes(started[0][index - 1]) ? `<${started[0][index - 1]}>` : arg));
    const saved = structuredClone(controller.state.params);
    if (drain) await host.drain();
    return { args: masked, procDir, controller, saved };
}
const flag = (args, name) => args[args.indexOf(name) + 1];
const PROFILES = ['cpu', 'dedicated', 'unified', 'metal'];

test('V2b: threads 2 on a 16-core fake puts --threads 2 on the launch of every profile, where no cap gives the engine default (the V5 control)', async (t) => {
    for (const profile of PROFILES) {
        const capped = await launchOf(t, { caps: { threads: 2 }, profile });
        const plain = await launchOf(t, { profile });
        const expected = defaultThreadsOf(profile, plain);
        assert.ok(expected > 2, `${profile}: the engine default (${expected}) is above the cap, so the cap is what is seen`);
        assert.equal(flag(plain.args, '--threads'), String(expected), `${profile}: with no caps the argv has the engine default`);
        assert.equal(flag(capped.args, '--threads'), '2', `${profile}: the cap`);
        // Nothing else in the argv moved.
        const without = (args) => args.filter((arg, index) => args[index - 1] !== '--threads');
        assert.deepEqual(without(capped.args), without(plain.args), profile);
    }
});

test('V2b: a cap above the engine default changes nothing, and with no caps the argv is the identity argv', async (t) => {
    for (const profile of PROFILES) {
        const plain = await launchOf(t, { profile });
        const roomy = await launchOf(t, { caps: { threads: 200 }, profile });
        assert.deepEqual(roomy.args, plain.args, profile);
    }
});

test('V2b: an explicit --param threads=8 under threads 2 is refused naming threads, and the same request with no caps launches with --threads 8', async (t) => {
    for (const profile of PROFILES) {
        const refused = await launchOf(t, { caps: { threads: 2 }, profile, params: { threads: 8 } });
        assert.ok(refused.error, `${profile}: refused`);
        assert.equal(refused.error.code, 'invalid_params');
        assert.match(refused.error.message, /threads/);
        assert.match(refused.error.message, /localLlm\.caps\.threads/);
        assert.match(refused.error.message, /\b8\b/);
        assert.match(refused.error.message, /\b2\b/);
        const plain = await launchOf(t, { profile, params: { threads: 8 } });
        assert.equal(flag(plain.args, '--threads'), '8', `${profile}: the control`);
        // An explicit value at or under the cap is the caller's own and is kept.
        const within = await launchOf(t, { caps: { threads: 2 }, profile, params: { threads: 1 } });
        assert.equal(flag(within.args, '--threads'), '1', profile);
    }
});

test('V2b: maxContext and maxParallel lower the default --ctx-size and -np and refuse an explicit value, where no caps give the defaults', async (t) => {
    for (const profile of PROFILES) {
        const flagFor = profile === 'metal' ? '-c' : '--ctx-size';
        const plain = await launchOf(t, { profile, params: {} });
        const plainCtx = Number(flag(plain.args, flagFor));
        assert.ok(plainCtx > 1024, `${profile}: the default context (${plainCtx}) is above the cap`);
        const capped = await launchOf(t, { caps: { maxContext: 1024 }, profile });
        assert.equal(flag(capped.args, flagFor), '1024', `${profile}: the default context is lowered to the cap`);
        const refused = await launchOf(t, { caps: { maxContext: 1024 }, profile, params: { ctxSize: 2048 } });
        assert.equal(refused.error?.code, 'invalid_params', profile);
        assert.match(refused.error.message, /localLlm\.caps\.maxContext/);
        assert.match(refused.error.message, /ctxSize/);
        assert.match(refused.error.message, /2048/);
        assert.match(refused.error.message, /1024/);
        const below = await launchOf(t, { caps: { maxContext: 4096 }, profile, params: { ctxSize: 1024 } });
        assert.equal(flag(below.args, flagFor), '1024', `${profile}: an explicit value under the cap is kept`);
        const control = await launchOf(t, { profile, params: { ctxSize: 2048 } });
        assert.equal(flag(control.args, flagFor), '2048', `${profile}: the control`);

        // parallel: the caller sets 2 slots; a cap of 1 refuses it, no cap passes it; the default of 1 is never above a cap of 1.
        const two = await launchOf(t, { profile, params: { parallel: 2 } });
        assert.equal(flag(two.args, '-np'), '2', `${profile}: the control`);
        const refusedParallel = await launchOf(t, { caps: { maxParallel: 1 }, profile, params: { parallel: 2 } });
        assert.equal(refusedParallel.error?.code, 'invalid_params', profile);
        assert.match(refusedParallel.error.message, /localLlm\.caps\.maxParallel/);
        assert.equal(flag((await launchOf(t, { caps: { maxParallel: 1 }, profile })).args, '-np'), '1', profile);
    }
});

test('maxParallel lowers a saved parallel and the context sized for it, leaving the saved set unchanged; an explicit context the lowered slots cannot hold is refused naming maxParallel', async (t) => {
    const home = scratch(t);
    const wanted = { ctxSize: 4 * CONTEXT, parallel: 4 };
    // An earlier Run with no caps saves four slots of the model's training context each.
    const earlier = await launchOf(t, { profile: 'cpu', modelId: 'slots', params: wanted, home, drain: true });
    assert.equal(flag(earlier.args, '-np'), '4');
    assert.equal(flag(earlier.args, '--ctx-size'), String(4 * CONTEXT));
    const key = 'slots|llama.cpp-cpu';
    assert.deepEqual({ ctxSize: earlier.saved[key].ctxSize, parallel: earlier.saved[key].parallel }, wanted);
    // The same home under maxParallel 1, a Run with no params: the saved values are defaults, so they are lowered, not refused.
    const capped = await launchOf(t, { caps: { maxParallel: 1 }, profile: 'cpu', modelId: 'slots', home });
    assert.equal(capped.controller.state.params[key].parallel, 4, 'the saved set was loaded');
    assert.equal(flag(capped.args, '-np'), '1');
    assert.equal(flag(capped.args, '--ctx-size'), String(CONTEXT), 'the context follows the slots down to the training context of one');
    assert.deepEqual(capped.saved[key], earlier.saved[key], 'the saved set is unchanged');
    // The model recommends four slots (a default); an explicit context for four is refused by the cap, not by the runner.
    const refused = await launchOf(t, { caps: { maxParallel: 1 }, profile: 'cpu', modelId: 'slots-recommended', params: { ctxSize: 2 * CONTEXT } });
    assert.equal(refused.error?.code, 'invalid_params');
    assert.match(refused.error.message, /localLlm\.caps\.maxParallel/);
    assert.ok(refused.error.message.includes(String(2 * CONTEXT)) && refused.error.message.includes(String(CONTEXT)), refused.error.message);
    // The controls: within the lowered slots it is accepted, and with no caps the four recommended slots hold it.
    assert.equal(flag((await launchOf(t, { caps: { maxParallel: 1 }, profile: 'cpu', modelId: 'slots-recommended', params: { ctxSize: CONTEXT } })).args, '--ctx-size'), String(CONTEXT));
    const open = await launchOf(t, { profile: 'cpu', modelId: 'slots-recommended', params: { ctxSize: 2 * CONTEXT } });
    assert.equal(flag(open.args, '-np'), '4');
    assert.equal(flag(open.args, '--ctx-size'), String(2 * CONTEXT));
});

// ---------------------------------------------------------------- the refusal reaches HTTP and the CLI

/** The stand-in runner with the llama-server names for the three parameters the count caps bound. */
function cappedAdapter() {
    const schema = deepFreeze({
        type: 'object', additionalProperties: false,
        properties: {
            ctxSize: { type: 'integer', minimum: 512, maximum: 65536, default: 4096 },
            parallel: { type: 'integer', minimum: 1, maximum: 8, default: 1 },
            threads: { type: ['integer', 'null'], minimum: 1, maximum: 256, default: null },
        },
    });
    return Object.freeze({ ...fakeAdapter(), paramSchema: schema, basicParams: Object.freeze(['ctxSize']),
        normalizeParams: (params, { model } = {}) => normalizeWith(schema, 'fake-gguf', params, model),
        describeContext: ({ ctxSize, parallel }) => ({ totalContext: ctxSize, perRequestContext: ctxSize, parallel, kvUnified: false }) });
}

test('a value over a count cap is a 400 invalid_params on POST /v1/local-llm/run and /preview, naming the cap; a value within it is accepted', async (t) => {
    assert.equal(statusFor('invalid_params'), 400);
    const proxy = await startProxy(t, { localLlm: { caps: { threads: 2, maxContext: 1024 } }, hostOptions: { adapter: cappedAdapter() } });
    const post = (route, params) => request(proxy.base, `/v1/local-llm/${route}`, { method: 'POST', body: { modelId: MODEL, runnerId: 'fake-gguf', params } });
    for (const route of ['run', 'preview']) {
        const refused = await post(route, { threads: 8 });
        assert.equal(refused.status, 400, `${route}: ${refused.text}`);
        assert.equal(refused.body.error.type, 'invalid_params');
        assert.match(refused.body.error.message, /localLlm\.caps\.threads/);
        assert.match(refused.body.error.message, /\b8\b.*\b2\b/);
        const context = await post(route, { ctxSize: 2048 });
        assert.equal(context.status, 400, `${route}: ${context.text}`);
        assert.match(context.body.error.message, /localLlm\.caps\.maxContext/);
    }
    // A value within the caps, and none at all (the defaults, ctxSize 4096 lowered to 1024), are accepted.
    assert.equal((await post('preview', { threads: 2, ctxSize: 1024 })).status, 200);
    const preview = await post('preview', undefined);
    assert.equal(preview.status, 200);
    assert.equal(preview.body.preview.params.ctxSize, 1024, 'the preview shows the capped default');
    assert.equal(preview.body.preview.params.threads, null);
    assert.equal((await post('run', { threads: 1 })).status, 202);
    // The control: with no caps the same request is not refused.
    const plain = await startProxy(t, { hostOptions: { adapter: cappedAdapter() } });
    const open = await request(plain.base, '/v1/local-llm/preview', { method: 'POST', body: { modelId: MODEL, runnerId: 'fake-gguf', params: { threads: 8, ctxSize: 2048 } } });
    assert.equal(open.status, 200, open.text);
    assert.equal(open.body.preview.params.threads, 8);
});

test('pworker models run and preview print the refusal naming the cap and exit with an error; the control with no caps runs', async (t) => {
    const proxy = await startProxy(t, { localLlm: { caps: { threads: 2 } }, hostOptions: { adapter: cappedAdapter() } });
    const verb = (target, args) => {
        const out = [], err = [];
        const io = { out: (text) => out.push(text), err: (text) => err.push(text), isTTY: false, home: scratch({ after() {} }), stdin: new PassThrough(), stdout: new PassThrough() };
        const client = createPworkerClient({ url: target.base, purpose: 'test:models-cli', autostart: false, env: {} });
        return runEngineVerb({ args: ['models', ...args], client, io, pollMs: 20, lineMs: 0 }).then((code) => ({ code, out: out.join(''), err: err.join('') }), (error) => ({ error, out: out.join(''), err: err.join('') }));
    };
    for (const name of ['run', 'preview']) {
        const refused = await verb(proxy, [name, MODEL, '--runner', 'fake-gguf', '--param', 'threads=8', ...(name === 'run' ? ['--detach'] : [])]);
        assert.ok(refused.error, `${name}: the verb failed`);
        assert.equal(refused.error.type ?? refused.error.code, 'invalid_params');
        assert.match(refused.error.message, /localLlm\.caps\.threads/);
        assert.match(refused.error.message, /\b8\b.*\b2\b/);
    }
    const plain = await startProxy(t, { hostOptions: { adapter: cappedAdapter() } });
    const open = await verb(plain, ['run', MODEL, '--runner', 'fake-gguf', '--param', 'threads=8', '--detach']);
    assert.equal(open.error, undefined);
    assert.equal(open.code, 0);
});

// ---------------------------------------------------------------- the gate identity

test('the gate identity carries the caps digest only when caps are set, stable for equal caps and different for different ones', async (t) => {
    const sha = loadSeedCatalog(SEED_CATALOG_FILE).find((model) => model.id === MODEL).sources.gguf.sha256;
    const plainIdentity = `local-llm:${MODEL}:llama.cpp-cpu:b11295:${sha}`;
    const identityOf = async (caps) => {
        const host = await capsHost(t, { caps, profile: 'cpu', machine: CPU_MACHINE });
        return createLocalLlmGate({ host, settings: {} }).identity(MODEL, 'llama.cpp-cpu');
    };
    const none = await identityOf(undefined);
    assert.equal(none, plainIdentity, 'byte-identical with no caps');
    assert.ok(!none.includes(':caps-'));
    const eight = await identityOf({ ram: '8GiB' });
    assert.equal(eight, `${plainIdentity}:caps-${capsDigest(parseCaps({ ram: '8GiB' }))}`);
    assert.equal(await identityOf({ ram: 8 * GIB }), eight, 'equal caps, equal identity');
    const nine = await identityOf({ ram: '9GiB' });
    assert.ok(nine.includes(':caps-'));
    assert.notEqual(nine, eight);
    assert.notEqual(await identityOf({ gpuMemory: '8GiB' }), eight);
});

test('capsLine names the caps in force in human units and is silent without them', () => {
    assert.equal(capsLine({ limits: { budget: { caps: { ram: 12884901888, gpuMemory: 512 * 1024 * 1024 }, mode: 'refuse' } } }), 'caps in force (localLlm.caps): ram 12 GiB, gpuMemory 512 MiB; refuse mode');
    assert.equal(capsLine({ limits: null }), null);
    assert.equal(capsLine({ limits: { budget: { caps: {}, mode: 'refuse' } } }), null);
    assert.equal(capsLine(undefined), null);
});

test('models status, a refused preview and a refused run print the caps line; with no caps the output is the same as before', async () => {
    const capped = { limits: { budget: { caps: { ram: 8 * GIB }, mode: 'refuse' } } };
    const refusedPreview = { preview: { admission: { status: 'incompatible', reason: 'Needs about 13 GiB' } } };
    const verb = async (args, overview, engineExtra = {}) => {
        const out = [], err = [];
        const engine = { status: async () => ({ phase: 'idle', ...(overview?.limits?.budget ? { caps: true } : {}) }), overview: async () => overview, preview: async () => refusedPreview, ...engineExtra };
        const code = await runEngineVerb({ args: ['models', ...args], client: { engine }, io: { out: (text) => out.push(text), err: (text) => err.push(text), isTTY: false } }).catch((error) => error);
        return { code, out: out.join(''), err: err.join('') };
    };
    const line = 'caps in force (localLlm.caps): ram 8 GiB; refuse mode\n';
    assert.equal((await verb(['status'], capped)).out, `phase idle\n${line}`);
    assert.equal((await verb(['status'], { limits: null })).out, 'phase idle\n');
    assert.equal((await verb(['status', '--json'], capped)).out, `${JSON.stringify({ phase: 'idle', caps: true }, null, 2)}\n`, 'json output is the status alone');
    assert.equal((await verb(['preview', 'x'], capped)).out, `x: incompatible (Needs about 13 GiB)\n${line}`);
    assert.equal((await verb(['preview', 'x'], { limits: null })).out, 'x: incompatible (Needs about 13 GiB)\n');
    const refusedRun = { run: async () => { throw Object.assign(new Error('Needs about 13 GiB'), { type: 'admission_incompatible' }); } };
    const run = await verb(['run', 'x'], capped, refusedRun);
    assert.equal(run.code.message, 'Needs about 13 GiB', 'the refusal still propagates');
    assert.equal(run.err, line);
    assert.equal((await verb(['run', 'x'], { limits: null }, refusedRun)).err, '');
    assert.equal((await verb(['status'], capped, { overview: async () => { throw new Error('down'); } })).out, 'phase idle\n', 'an overview that fails never fails the verb');
});

// ---------------------------------------------------------------- the cap bounds the need, not the host reserve or the floor

test('a ram cap leaves the cpu floor and host reserve those of the physical machine, and a model that fits both is admitted', async (t) => {
    const capped = await capsHost(t, { caps: { ram: '8GiB' }, profile: 'cpu', machine: CPU_MACHINE });
    const plain = await capsHost(t, { profile: 'cpu', machine: CPU_MACHINE });
    const withCap = await previewOf(capped, 'cpu', 'small');
    const without = await previewOf(plain, 'cpu', 'small');
    assert.equal(withCap.status, 'ok', withCap.reason);
    assert.equal(withCap.estimate.floorBytes, cpuFloorBytes(64 * GIB), 'the guard floor is the physical machine\'s');
    assert.equal(withCap.estimate.hostReserveBytes, cpuHostReserveBytes(64 * GIB));
    assert.equal(withCap.estimate.floorBytes, without.estimate.floorBytes);
    assert.equal(withCap.estimate.hostReserveBytes, without.estimate.hostReserveBytes);
    assert.equal(withCap.estimate.poolBytes, 8 * GIB, 'the pool shown is the capped one');
    const limits = (await (await capped.controller()).overview()).limits;
    assert.equal(limits.floorBytes, cpuFloorBytes(64 * GIB), 'the overview shows the physical floor too');
    assert.equal(limits.hostReserveBytes, cpuHostReserveBytes(64 * GIB));
    assert.equal(limits.poolBytes, 8 * GIB);
});

test('a ram cap on a machine with little free memory still refuses for now: the physical floor applies to what is available', async (t) => {
    // 9 GiB free of 64: the floor is 6.4 GiB, so only 2.6 GiB can be used now whatever the cap says; the 2 GiB model needs 3 GiB.
    const busy = { ...CPU_MACHINE, memory: { totalBytes: 64 * GIB, availableBytes: 9 * GIB } };
    const capped = await capsHost(t, { caps: { ram: '40GiB' }, profile: 'cpu', machine: busy });
    const verdict = await previewOf(capped, 'cpu', 'small');
    assert.equal(verdict.status, 'insufficient-now', verdict.reason);
    assert.equal(verdict.estimate.floorBytes, cpuFloorBytes(64 * GIB));
    // With a cap of 8 GiB the same machine and model give the same verdict, for the same reason.
    const tighter = await capsHost(t, { caps: { ram: '8GiB' }, profile: 'cpu', machine: busy });
    assert.equal((await previewOf(tighter, 'cpu', 'small')).status, 'insufficient-now');
});

test('on the metal profile a ram cap leaves the floor and host reserve those of the physical Mac', () => {
    const macMemory = { totalBytes: 16 * GIB, availableBytes: 12 * GIB, pressureLevel: 1 };
    const snapshot = { gpu: { available: true, backend: 'metal', name: 'Apple M1', memoryModel: 'unified', deviceCapBytes: 12 * GIB }, memory: macMemory, disk: DISK, cpus: 8, cores: 4 };
    const model = loadSeedCatalog(SEED_CATALOG_FILE).find((entry) => entry.id === 'qwen2.5-0.5b-instruct-q4_k_m');
    const runner = getRunner('llama.cpp');
    const verdictWith = (limits) => admit({
        runner, model, source: model.sources.gguf, snapshot, profile: 'metal', limits,
        params: runner.normalizeParams({}, { model, profile: 'metal' }),
    });
    const plain = verdictWith(IDENTITY_LIMITS);
    const capped = verdictWith(defineCapLimits(parseCaps({ ram: '8GiB' })));
    assert.equal(plain.status, 'ok', plain.reason);
    assert.equal(capped.status, 'ok', capped.reason);
    assert.equal(capped.estimate.floorBytes, cpuFloorBytes(16 * GIB));
    assert.equal(capped.estimate.hostReserveBytes, cpuHostReserveBytes(16 * GIB));
    assert.equal(capped.estimate.floorBytes, plain.estimate.floorBytes);
    assert.equal(capped.estimate.hostReserveBytes, plain.estimate.hostReserveBytes);
    assert.equal(capped.estimate.poolBytes, 8 * GIB);
    // A cap under the model's need refuses it, and the reason is about the capacity.
    const tight = verdictWith(defineCapLimits(parseCaps({ ram: '1GiB' })));
    assert.equal(tight.status, 'incompatible', tight.reason);
    assert.equal(tight.reasonCode, 'metal_capacity');
});

test('a unified GPU\'s gpuMemory bounds the need only: the 16 GiB host reserve comes from the 128 GiB machine, not from the cap', async (t) => {
    // The unified policy\'s own fixed need (a prompt cache of 8 GiB, the runtime and the buffers) is about 10.4 GiB even for a tiny
    // model, so no model has a need under 8 GiB here: 12 GiB is the smallest round cap that admits one.
    const big = { ...UNIFIED_MACHINE, memory: { totalBytes: 128 * GIB, availableBytes: 120 * GIB } };
    const host = await capsHost(t, { caps: { gpuMemory: '12GiB' }, profile: 'unified', machine: big });
    const fits = await previewOf(host, 'unified', 'tiny');
    assert.equal(fits.status, 'ok', fits.reason);
    assert.ok(fits.estimate.unifiedBytes < 12 * GIB && fits.estimate.unifiedBytes > 10 * GIB, `needs ${fits.estimate.unifiedBytes}`);
    assert.equal(fits.estimate.poolBytes, 12 * GIB, 'the pool shown is the capped one');
    const over = await previewOf(host, 'unified', 'small');
    assert.equal(over.status, 'incompatible', over.reason);
    assert.ok(over.estimate.unifiedBytes > 12 * GIB, `needs ${over.estimate.unifiedBytes}`);
    assert.match(over.reason, /of the 12\.0+ GiB/, 'the capped figure is in the reason');
    const under = await capsHost(t, { caps: { gpuMemory: '8GiB' }, profile: 'unified', machine: big });
    assert.equal((await previewOf(under, 'unified', 'tiny')).status, 'incompatible', 'a cap under the smallest need refuses it');
    const plain = await capsHost(t, { profile: 'unified', machine: big });
    assert.equal((await previewOf(plain, 'unified', 'small')).status, 'ok', 'and with no cap the larger model fits this machine');
    // The floor still applies to what is available now: 120 GiB are free, so the cap alone decided above.
    const busy = await capsHost(t, { caps: { gpuMemory: '48GiB' }, profile: 'unified', machine: { ...big, memory: { totalBytes: 128 * GIB, availableBytes: 14 * GIB } } });
    assert.equal((await previewOf(busy, 'unified', 'tiny')).status, 'insufficient-now', 'the 8 GiB floor is taken from the physical available memory');
});

test('gpuMemory is refused on macOS, naming ram, and a cpu overview does not publish it', async (t) => {
    const refused = createLocalLlmHost({ config: { localLlm: { caps: { gpuMemory: '8GiB' } } }, home: scratch(t), env: {}, platformName: 'darwin', arch: 'arm64' });
    assert.equal(refused.caps, null);
    await assert.rejects(refused.controller(), (error) => error.code === 'invalid_caps' && error.details.key === 'gpuMemory' && /use localLlm\.caps\.ram/.test(error.message));
    assert.equal(createLocalLlmHost({ config: { localLlm: { caps: { ram: '8GiB' } } }, home: scratch(t), env: {}, platformName: 'darwin', arch: 'arm64' }).capsRefusal, null, 'ram is fine on a Mac');
    const host = await capsHost(t, { caps: { ram: '8GiB', gpuMemory: '4GiB' }, profile: 'cpu', machine: CPU_MACHINE });
    assert.deepEqual((await (await host.controller()).overview()).limits.budget, { caps: { ram: 8 * GIB }, mode: 'refuse' });
});

test('a bad caps config is a 503, the gate says so instead of "no model is ready", and the status route flags caps only when they are set', async (t) => {
    assert.equal(statusFor('invalid_caps'), 503);
    const bad = await startHost(t, { config: { localLlm: { caps: { ram: 0 } } } });
    const verdict = await createLocalLlmGate({ host: bad.host, settings: {} }).ensure({ model: MODEL, runner: 'fake-gguf' });
    assert.equal(verdict.ok, false);
    assert.match(verdict.reason, /localLlm\.caps\.ram/);
    assert.doesNotMatch(verdict.reason, /pworker models run/);
    const proxy = await startProxy(t, { localLlm: { caps: { ram: '8GiB' } } });
    assert.equal((await request(proxy.base, '/v1/local-llm/status')).body.caps, true);
    const plain = await startProxy(t);
    assert.equal(Object.hasOwn((await request(plain.base, '/v1/local-llm/status')).body, 'caps'), false, 'no caps: the status is as it was');
    const badProxy = await startProxy(t, { localLlm: { caps: { ram: 0 } } });
    const refused = await request(badProxy.base, '/v1/local-llm/status');
    assert.equal(refused.body.refusal.code, 'invalid_caps');
});
