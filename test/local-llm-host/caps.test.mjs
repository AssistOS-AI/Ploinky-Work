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

import { loadSeedCatalog } from '../../lib/local-llm/controller/catalog.mjs';
import { LocalLlmError, defineLimits, IDENTITY_LIMITS } from '../../lib/local-llm/index.mjs';
import { capsDigest, defineCapLimits, parseCaps } from '../../lib/local-llm-host/caps.mjs';
import { createLocalLlmGate } from '../../lib/local-llm-host/gate.mjs';
import { admit } from '../../lib/local-llm/controller/admission.mjs';
import { cpuFloorBytes, cpuHostReserveBytes } from '../../lib/local-llm/controller/profiles.mjs';
import { getRunner } from '../local-llm/fixtures/host.mjs';
import { SEED_CATALOG_FILE, createLocalLlmHost } from '../../lib/local-llm-host/host.mjs';
import { LAUNCH_SCHEMA } from '../../lib/local-llm-host/launch.mjs';
import { capsLine, runEngineVerb } from '../../lib/pworker/models-cli.mjs';
import { statusFor } from '../../lib/local-llm-host/routes.mjs';
import { GIB, MODEL, freePort, scratch } from './fixtures/engine.mjs';
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
const SEED = Object.entries(SIZES).map(([id, gib]) => modelOf(id, gib));

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
        [{ threads: 4 }, 'threads'], [{ maxContext: 4096 }, 'maxContext'], [{ maxParallel: 2 }, 'maxParallel'], [{ mode: 'fit' }, 'mode'],
        [{ ram: '8GiB', mode: 'strict' }, 'mode'], [{ gpuMemory: 12 }, 'gpuMemory'], [{ ram: 1048575 }, 'ram'], [{ ram: '512KiB' }, 'ram'],
    ];
    for (const [section, key] of cases) {
        assert.throws(() => parseCaps(section), (error) => error instanceof LocalLlmError && error.code === 'invalid_caps'
            && error.details.key === key && error.message.includes(key), JSON.stringify(section));
    }
    for (const key of ['threads', 'maxContext', 'maxParallel']) {
        assert.throws(() => parseCaps({ [key]: 2 }), (error) => /not supported yet/.test(error.message), key);
    }
    assert.throws(() => parseCaps({ mode: 'fit' }), /not supported yet/);
    assert.throws(() => parseCaps([]), (error) => error.code === 'invalid_caps');
    assert.throws(() => parseCaps('8GiB'), (error) => error.code === 'invalid_caps');
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

test('capsDigest is stable for equal caps and differs for different ones', () => {
    const digest = capsDigest(parseCaps({ ram: '8GiB' }));
    assert.match(digest, /^[0-9a-f]{12}$/);
    assert.equal(capsDigest(parseCaps({ ram: 8 * GIB })), digest, 'the same bytes, however they were written');
    assert.equal(capsDigest(parseCaps({ ram: '8GiB', mode: 'refuse' })), digest);
    assert.notEqual(capsDigest(parseCaps({ ram: '9GiB' })), digest);
    assert.notEqual(capsDigest(parseCaps({ gpuMemory: '8GiB' })), digest, 'the key counts, not only the number');
    assert.notEqual(capsDigest(parseCaps({ ram: '8GiB', gpuMemory: '8GiB' })), digest);
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

test('an unsupported key is refused the same way, and the host with no caps has the identity limits', async (t) => {
    const host = await capsHost(t, { caps: { threads: 4 }, profile: 'cpu', machine: CPU_MACHINE });
    await assert.rejects(host.controller(), (error) => error.code === 'invalid_caps' && error.details.key === 'threads' && /not supported yet/.test(error.message));
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
