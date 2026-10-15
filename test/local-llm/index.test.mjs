// The public entry (lib/local-llm/index.mjs): its export surface, and a controller composed from a host alone.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import * as engine from '../../lib/local-llm/index.mjs';
import { createController } from '../../lib/local-llm/controller/deployments.mjs';
import { DRAIN_QUEUE_WAIT_MS, DRAIN_RUNNER_GRACE_MS, HARDWARE_QUERY_REAP_MS } from '../../lib/local-llm/drainBudget.mjs';
import { SEED_CATALOG_FILE } from './fixtures/seedCatalog.mjs';
import { IMAGE_LOCK } from './fixtures/sharedFixtures.mjs';

const GIB = 1024 ** 3;
const DEDICATED = Object.freeze({
    gpu: { available: true, name: 'RTX', memoryModel: 'dedicated', totalBytes: 6 * GIB, freeBytes: 6 * GIB, usedBytes: 0, processes: [] },
    memory: { totalBytes: 32 * GIB, availableBytes: 28 * GIB },
    disk: { freeBytes: 400 * GIB },
});

const CPU_ONLY = Object.freeze({
    gpu: { available: false, state: 'absent' },
    memory: { totalBytes: 32 * GIB, availableBytes: 28 * GIB },
    disk: { freeBytes: 400 * GIB },
    cores: 8,
});

function tempDir(t) {
    const dir = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), 'local-llm-index-'));
    t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
    return dir;
}

const specOf = (dir, extra = {}) => ({ dataDir: path.join(dir, 'data'), runRoot: path.join(dir, 'run'), seedCatalogFile: SEED_CATALOG_FILE, ...extra });

test('the public entry exports exactly the documented surface', () => {
    assert.deepEqual(Object.keys(engine).sort(), [
        'DEFAULT_MESSAGES', 'ENGINE_DRAIN', 'IDENTITY_LIMITS', 'LIMIT_HOOKS', 'LMSTUDIO_SWITCH', 'LocalLlmError', 'MESSAGE_KEYS',
        'OBSERVATION_ID_RE', 'RUNNER_IDS', 'VLLM_UNIFIED_SWITCH', 'computeDiagnostic', 'createController', 'createLocalLlmController',
        'createObservationClock', 'createRunnerInstaller', 'createRunners', 'createStartupObservations', 'createStateStore', 'defineHost',
        'defineLimits', 'defineMessages', 'linuxPlatform', 'loadRunnerLocks', 'loadSeedCatalog', 'readSnapshot', 'runnerSummaries',
        'serializeError', 'validateModel',
    ]);
    for (const [name, value] of Object.entries(engine)) {
        assert.ok(['function', 'object', 'string'].includes(typeof value), name);
        assert.notEqual(value, undefined, name);
    }
    assert.equal(engine.createController, createController);
    assert.deepEqual({ ...engine.ENGINE_DRAIN }, { queueWaitMs: DRAIN_QUEUE_WAIT_MS, hardwareQueryReapMs: HARDWARE_QUERY_REAP_MS, runnerGraceMs: DRAIN_RUNNER_GRACE_MS });
    assert.ok(Object.isFrozen(engine.ENGINE_DRAIN));
    assert.equal(engine.LMSTUDIO_SWITCH, 'LOCAL_LLM_LMSTUDIO');
    assert.equal(engine.VLLM_UNIFIED_SWITCH, 'LOCAL_LLM_VLLM_UNIFIED');
    assert.match('abcdefgh', engine.OBSERVATION_ID_RE);
});

test('RUNNER_IDS lists the runners createRunners builds, in order', () => {
    const host = engine.defineHost({ dataDir: '/x/data', runRoot: '/x/run' });
    assert.deepEqual([...engine.RUNNER_IDS], Object.keys(engine.createRunners({ host })));
    assert.ok(Object.isFrozen(engine.RUNNER_IDS));
});

test('a controller composes from a host alone: plain specs are validated, the gateway name is null and an invalid input is refused where the controller is made', async (t) => {
    const dir = tempDir(t);
    const controller = engine.createLocalLlmController({ host: specOf(dir), env: { PATH: '/usr/bin' }, snapshot: async () => structuredClone(DEDICATED) });
    t.after(() => controller.drain());
    const overview = await controller.overview();
    assert.equal(overview.gatewayModel, null);
    assert.ok(overview.models.length > 0);
    assert.deepEqual(overview.runners.map((runner) => runner.id), engine.RUNNER_IDS.filter((id) => overview.runners.some((runner) => runner.id === id)));
    assert.throws(() => controller.chatTarget(), { code: 'not_ready', message: engine.DEFAULT_MESSAGES.notReady() });
    // No host, a host with an unknown key and a message that is not a function are refused where the controller is made.
    assert.throws(() => engine.createLocalLlmController({}), { code: 'invalid_host' });
    assert.throws(() => engine.createLocalLlmController({ host: { ...specOf(dir), extra: 1 } }), { code: 'invalid_host' });
    assert.throws(() => engine.createLocalLlmController({ host: specOf(dir), messages: { notReady: 'text' } }), { code: 'invalid_messages' });
    assert.throws(() => engine.createLocalLlmController({ host: specOf(dir), limits: { nope: () => null } }), { code: 'invalid_limits' });
});

test('the gateway name, messages and limits are the caller\'s, and the environment is handed over by reference', async (t) => {
    const dir = tempDir(t);
    const env = { PATH: '/usr/bin' };
    const seen = [];
    const messages = engine.defineMessages({ notReady: () => 'custom not ready' });
    const limits = engine.defineLimits({ overview: (context) => { seen.push(context); return null; } });
    const controller = engine.createLocalLlmController({
        host: engine.defineHost(specOf(dir)), messages, limits, gatewayModelName: 'gw/model/default', env, snapshot: async () => structuredClone(DEDICATED),
    });
    t.after(() => controller.drain());
    assert.equal((await controller.overview()).gatewayModel, 'gw/model/default');
    assert.throws(() => controller.chatTarget(), { code: 'not_ready', message: 'custom not ready' });
    assert.ok(seen.length > 0);
    assert.ok(seen.every((context) => context.env === env), 'the limits see the very object the caller passed');
});

test('the runner locks come from the host\'s files, and the host lock is asked for this architecture', async (t) => {
    const dir = tempDir(t);
    const imageLock = path.join(dir, 'image.lock.json');
    const agentLock = path.join(dir, 'agent.lock.json');
    fs.writeFileSync(imageLock, JSON.stringify(IMAGE_LOCK));
    fs.writeFileSync(agentLock, JSON.stringify({
        schema: 'local-llm.runners-lock/v1',
        runners: {
            'llama.cpp-cpu': {
                version: '1.0.0', kind: 'archive', licence: { name: 'MIT', url: 'https://example.org/LICENSE' },
                files: [{ name: 'cpu-1.0.0.tar.gz', url: 'https://github.com/example/r/releases/download/1.0.0/cpu-1.0.0.tar.gz', size: 10, sha256: 'a'.repeat(64), strip: 0 }],
                check: { executable: 'llama-server' },
            },
        },
    }));
    const asked = [];
    const host = engine.defineHost(specOf(dir, {
        imageRunnerLockFile: imageLock, uv: '/usr/bin/uv', python: '/usr/bin/python3',
        agentRunnerLockFile: (arch) => { asked.push(arch); return arch === 'arm64' ? agentLock : null; },
    }));
    const installs = async (hostArch) => {
        const controller = engine.createLocalLlmController({ host, hostArch, env: {}, snapshot: async () => structuredClone(CPU_ONLY) });
        t.after(() => controller.drain());
        const overview = await controller.overview();
        return overview.runners.filter((runner) => runner.install).map((runner) => runner.id).sort();
    };
    const onArm = await installs('arm64');
    const onX64 = await installs('x64');
    assert.deepEqual(asked, ['arm64', 'x64']);
    assert.ok(onArm.includes('vllm') && onX64.includes('vllm'), 'the image lock applies on both');
    assert.ok(onArm.includes('llama.cpp-cpu'), 'the host lock of this architecture applies');
    assert.equal(onX64.includes('llama.cpp-cpu'), false, 'a host with no lock for this architecture offers none');
});

test('seams win over what the entry composes, and a seam skips the work that would have made it', async (t) => {
    const dir = tempDir(t);
    const host = engine.defineHost(specOf(dir, {
        imageRunnerLockFile: path.join(dir, 'missing.lock.json'),
        agentRunnerLockFile: () => { throw new Error('the host lock must not be asked when the locks are injected'); },
    }));
    const runners = engine.createRunners({ host });
    const calls = [];
    const controller = engine.createLocalLlmController({
        host, runners, runnerLocks: { runners: {}, clashes: [], ignored: [] }, hostArch: 'riscv64', env: {},
        readHardwareSnapshot: async (options) => { calls.push(options); return structuredClone(DEDICATED); },
    });
    t.after(() => controller.drain());
    await controller.overview();
    assert.ok(calls.length > 0, 'the injected snapshot reader was used');
    const seedSeam = engine.createLocalLlmController({ host, runnerLocks: { runners: {}, clashes: [], ignored: [] }, seedCatalog: [], env: {}, snapshot: async () => structuredClone(DEDICATED) });
    t.after(() => seedSeam.drain());
    assert.deepEqual((await seedSeam.overview()).models, []);
});
