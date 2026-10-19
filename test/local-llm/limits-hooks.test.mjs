// The limits hooks (slice C of the engine move): the identity default, and for each hook the proof that the engine calls it
// at its touchpoint with the documented inputs and uses what it returns. A recording fake plug-in stands in for the
// deployment that would bind real limits; every assertion about a result is about what the engine does with the fake's
// answer. The guard hooks are driven through the pooled guard in apu-profile.test.mjs.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { admissionResult, admit, admitLlamaServer, admitLmStudio, admitOllama, admitVllm } from '../../lib/local-llm/controller/admission.mjs';
import { FIT_CONTEXT_FLOOR, FIT_KEYS, MAX_FIT_STEPS, fitAdmission } from '../../lib/local-llm/controller/fit.mjs';
import { loadSeedCatalog, validateModel } from '../../lib/local-llm/controller/catalog.mjs';
import { cpuDefaultThreads, defaultThreads, performanceCoreCount, physicalCoreCount, readSnapshot } from '../../lib/local-llm/controller/hardware.mjs';
import { UNIFIED, admitApuLlamaServer, cpuFloorBytes, cpuHostReserveBytes } from '../../lib/local-llm/controller/profiles.mjs';
import { linuxPlatform } from '../../lib/local-llm/host.mjs';
import { LocalLlmError } from '../../lib/local-llm/errors.mjs';
import { IDENTITY_LIMITS, LIMIT_HOOKS, defineLimits } from '../../lib/local-llm/limits.mjs';
import { createRunners } from '../../lib/local-llm/runners/index.mjs';
import { LLAMA_SERVER_PARAM_SCHEMA } from '../../lib/local-llm/runners/llamaServer.mjs';
import { createStateStore } from '../../lib/local-llm/controller/stateStore.mjs';
import { SEED_CATALOG_FILE } from './fixtures/seedCatalog.mjs';
import { HOST, PLOINKY_MESSAGES, RUNNERS, agentLikeHost, createController, getRunner } from './fixtures/host.mjs';
import { ggufBytes, modelPairs } from './gguf-fixture.mjs';
import { cpuFs } from './fixtures/sharedFixtures.mjs';

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;
const HOOKS = LIMIT_HOOKS.filter((name) => name !== 'vllmQualification');

/** A plug-in whose hooks answer as the identity default unless `answers` says otherwise, and record every call. */
function recording(answers = {}, { qualification } = {}) {
    const calls = [];
    const hooks = {};
    for (const name of HOOKS) {
        hooks[name] = (...args) => {
            calls.push({ name, args });
            const answer = answers[name];
            return typeof answer === 'function' ? answer(...args) : name in answers ? answer : IDENTITY_LIMITS[name](...args);
        };
    }
    if (qualification) {
        hooks.vllmQualification = (...args) => { calls.push({ name: 'vllmQualification', args }); return qualification(...args); };
    }
    return { limits: defineLimits(hooks), calls, named: (name) => calls.filter((call) => call.name === name) };
}

const ENOENT = () => Object.assign(new Error('absent'), { code: 'ENOENT' });

// ---------------------------------------------------------------- the contract and the identity default

test('limits: the default is the identity of every hook, and the value is frozen', () => {
    assert.deepEqual(LIMIT_HOOKS, ['capCores', 'capThreads', 'capParams', 'fitParams', 'observeSnapshot', 'decorateSnapshot', 'memoryPool', 'refuseProfile', 'admit', 'effective',
        'refuseResourceRequest', 'overview', 'runnerEnvironment', 'guardSample', 'guardBreach', 'vllmQualification']);
    assert.ok(Object.isFrozen(IDENTITY_LIMITS));
    assert.equal(IDENTITY_LIMITS.capCores(7, { fsApi: {} }), 7);
    const snapshot = { gpu: {}, memory: { totalBytes: 1 } };
    assert.equal(IDENTITY_LIMITS.capThreads(9, { profile: 'cpu' }), 9);
    const params = { ctxSize: 1 };
    assert.equal(IDENTITY_LIMITS.capParams({ runnerId: 'llama.cpp', profile: 'cpu', params, explicit: [], model: {} }), params);
    assert.equal(IDENTITY_LIMITS.fitParams({ runnerId: 'llama.cpp', profile: 'dedicated' }), null);
    assert.equal(IDENTITY_LIMITS.observeSnapshot({ fsApi: {}, env: {} }), null);
    assert.equal(IDENTITY_LIMITS.decorateSnapshot({ snapshot, fsApi: {}, env: {} }), snapshot);
    assert.equal(IDENTITY_LIMITS.memoryPool({ totalBytes: 1, availableBytes: 1 }, snapshot, { amount: String }), null);
    assert.equal(IDENTITY_LIMITS.refuseProfile({ stage: 'admission', profile: 'apu', snapshot, refuse: () => 'x' }), null);
    for (const stage of ['before-policy', 'vulkan', 'gpu']) assert.equal(IDENTITY_LIMITS.admit({ stage, profile: 'dedicated', snapshot, env: {} }), null);
    assert.equal(IDENTITY_LIMITS.effective({ snapshot }).memory, snapshot.memory);
    assert.deepEqual(IDENTITY_LIMITS.effective({ snapshot: null }), { memory: {} });
    assert.equal(IDENTITY_LIMITS.refuseResourceRequest({ policy: 'vllm' }), null);
    assert.equal(IDENTITY_LIMITS.overview({ snapshot, env: {} }), null);
    const adapterEnvironment = { A: '1' };
    assert.equal(IDENTITY_LIMITS.runnerEnvironment({ env: {}, profile: 'dedicated', adapterEnvironment }), adapterEnvironment);
    assert.equal(IDENTITY_LIMITS.guardSample(), null);
    assert.equal(IDENTITY_LIMITS.guardBreach(null, { check: 'watchable', profile: 'apu' }), null);
    assert.equal(IDENTITY_LIMITS.guardBreach({ watched: true, usedBytes: 1 }, { check: 'headroom', memory: {}, available: 1 }), null, 'no stop, no figure, no unreadable');
    assert.equal(IDENTITY_LIMITS.vllmQualification, null);
});

test('defineLimits: unknown hooks and non-functions throw with the key; it is idempotent', () => {
    const invalid = (key) => (error) => error instanceof LocalLlmError && error.code === 'invalid_limits' && error.details?.key === key;
    assert.throws(() => defineLimits({ bogus: () => null }), invalid('bogus'));
    assert.throws(() => defineLimits({ capCores: 3 }), invalid('capCores'));
    assert.throws(() => defineLimits({ vllmQualification: 'yes' }), invalid('vllmQualification'));
    assert.throws(() => defineLimits(null), invalid('(limits)'));
    assert.throws(() => defineLimits([]), invalid('(limits)'));
    const capCores = (count) => count - 1;
    const own = defineLimits({ capCores, vllmQualification: null });
    assert.equal(own.capCores, capCores);
    assert.equal(own.memoryPool, IDENTITY_LIMITS.memoryPool);
    assert.ok(Object.isFrozen(own));
    assert.deepEqual({ ...defineLimits(own) }, { ...own });
});

// ---------------------------------------------------------------- capCores

const noFiles = { readFileSync() { throw ENOENT(); } };

test('capCores: physicalCoreCount hands the found count and its reader to the hook and returns what the hook answers', () => {
    const { limits, calls } = recording({ capCores: (count) => count - 3 });
    assert.equal(physicalCoreCount({ fsApi: noFiles, availableParallelism: () => 8, limits }), 5);
    assert.equal(calls.length, 1);
    assert.equal(calls[0].args[0], 8);
    assert.equal(calls[0].args[1].fsApi, noFiles);
    assert.equal(physicalCoreCount({ fsApi: noFiles, availableParallelism: () => 8 }), 8);
});

test('capCores: performanceCoreCount calls it with the performance class size, and not when there is no distinct class', () => {
    const capacities = (values) => ({ readFileSync(file) {
        if (file === '/proc/self/status') return 'Cpus_allowed_list:\t0-3\n';
        const match = /cpu(\d)\/cpu_capacity$/.exec(file);
        if (match) return String(values[Number(match[1])]);
        throw ENOENT();
    } });
    const mixed = capacities([1024, 1024, 700, 700]);
    const { limits, calls } = recording({ capCores: () => 1 });
    assert.equal(performanceCoreCount({ fsApi: mixed, limits }), 1);
    assert.equal(calls.length, 1);
    assert.equal(calls[0].args[0], 2);
    assert.equal(calls[0].args[1].fsApi, mixed);
    assert.equal(performanceCoreCount({ fsApi: mixed }), 2);
    const even = recording({ capCores: () => 1 });
    assert.equal(performanceCoreCount({ fsApi: capacities([1024, 1024, 1024, 1024]), limits: even.limits }), null);
    assert.equal(even.calls.length, 0);
});

const launchOf = (runner, profile = 'dedicated') => runner.buildLaunch({ artifactPath: '/m/q.gguf', params: {}, port: 18080, apiKey: 'k'.repeat(43),
    model: { id: 'm', displayName: 'M' }, runnerDir: '/x/run/llama.cpp/b1', profile });
const threadsOf = ({ args }) => args[args.indexOf('--threads') + 1];

test('capCores: createRunners gives the limits to the thread defaults of llama.cpp, ik_llama.cpp and the Vulkan and CPU builds', () => {
    const { limits, calls } = recording({ capCores: () => 40 });
    const runners = createRunners({ host: HOST, messages: PLOINKY_MESSAGES, limits });
    const physical = String(defaultThreads(physicalCoreCount()));
    for (const id of ['llama.cpp', 'ik_llama.cpp']) {
        assert.equal(threadsOf(launchOf(runners[id])), '38', `${id} with the capped 40 cores`);
        assert.equal(threadsOf(launchOf(RUNNERS[id])), physical, `${id} with the physical cores`);
    }
    assert.ok(calls.every((call) => call.name === 'capCores' || call.name === 'capThreads'));
    assert.ok(calls.filter((call) => call.name === 'capCores').length >= 2);
});

test('capCores: the performance class of the unified and CPU thread defaults is capped by the hook too', (t) => {
    // Four CPUs, two of them a higher-capacity class: the reader the runner uses is the real one, so the facts are mocked.
    const facts = (file) => {
        if (file === '/proc/self/status') return 'Cpus_allowed_list:\t0-3\n';
        const match = /cpu(\d)\/cpu_capacity$/.exec(file);
        if (match) return String([1024, 1024, 700, 700][Number(match[1])]);
        return null;
    };
    const read = fs.readFileSync;
    t.mock.method(fs, 'readFileSync', (file, ...args) => {
        const text = facts(String(file));
        if (text !== null) return text;
        return read(file, ...args);
    });
    const { limits, calls } = recording({ capCores: (count) => count + 30 });
    const runners = createRunners({ host: HOST, messages: PLOINKY_MESSAGES, limits });
    assert.equal(threadsOf(launchOf(runners['llama.cpp'], 'unified')), '32', 'the 2 performance cores, capped to 32');
    assert.equal(threadsOf(launchOf(RUNNERS['llama.cpp'], 'unified')), '2', 'the 2 performance cores');
    assert.equal(calls.filter((call) => call.args[0] === 2).length >= 1, true);
});

// ---------------------------------------------------------------- decorateSnapshot

test('decorateSnapshot: readSnapshot hands over the physical snapshot in key order, its reader and env, and publishes the result', async (t) => {
    const env = { MARK: '1' };
    const fsApi = cpuFs();
    const decorated = { at: 'decorated' };
    const { limits, calls } = recording({ decorateSnapshot: ({ snapshot }) => { snapshot.gpu = { available: false, state: 'unusable' }; snapshot.extra = 1; return decorated; } });
    const options = { dataDir: '/tmp', fsApi, env, statfs: async () => ({ bsize: 4096, blocks: 100, bavail: 50 }) };
    assert.equal(await readSnapshot({ ...options, limits }), decorated);
    const decorations = calls.filter((call) => call.name === 'decorateSnapshot');
    assert.equal(decorations.length, 1);
    const [{ snapshot, fsApi: givenFs, env: givenEnv }] = decorations[0].args;
    assert.deepEqual(Object.keys(snapshot), ['at', 'gpu', 'memory', 'disk', 'cpus', 'cores', 'extra']);
    assert.equal(givenFs, fsApi);
    assert.equal(givenEnv, env);
    assert.equal(snapshot.memory.totalBytes, 6291456 * 1024);
    // The default publishes the physical keys alone, and a reader that was not given gets the real one.
    const plain = await readSnapshot(options);
    assert.deepEqual(Object.keys(plain), ['at', 'gpu', 'memory', 'disk', 'cpus', 'cores']);
    const facts = cpuFs();
    const read = fs.readFileSync;
    t.mock.method(fs, 'readFileSync', (file, ...args) => String(file).startsWith('/proc/') ? facts.readFileSync(file) : read(file, ...args));
    const defaulted = recording();
    await readSnapshot({ dataDir: '/tmp', env, limits: defaulted.limits, fsApi: undefined, statfs: options.statfs });
    assert.equal(defaulted.calls.find((call) => call.name === 'decorateSnapshot').args[0].fsApi, fs);
});

test('decorateSnapshot: a key the plug-in assigns to an existing property keeps its place, and a new key is appended last', async () => {
    const { limits } = recording({ decorateSnapshot: ({ snapshot }) => { snapshot.gpu = { available: false }; snapshot.late = null; return snapshot; } });
    const snapshot = await readSnapshot({ dataDir: '/tmp', fsApi: cpuFs(), env: {}, limits, statfs: async () => ({ bsize: 1, blocks: 1, bavail: 1 }) });
    assert.deepEqual(Object.keys(snapshot), ['at', 'gpu', 'memory', 'disk', 'cpus', 'cores', 'late']);
});

test('observeSnapshot: the engine calls it after the disk read and before the memory read and the core count, and hands the result to decorateSnapshot', async () => {
    const events = [];
    const env = { MARK: '2' };
    const base = cpuFs();
    const fsApi = {
        readFileSync(file, ...rest) { events.push(`read:${file}`); return base.readFileSync(file, ...rest); },
        readdirSync(...args) { return base.readdirSync(...args); },
    };
    const observation = { token: 'observed' };
    const { limits, calls } = recording({
        observeSnapshot: () => { events.push('observeSnapshot'); return observation; },
        capCores: (count) => { events.push('capCores'); return count; },
        decorateSnapshot: ({ snapshot }) => { events.push('decorateSnapshot'); return snapshot; },
    });
    const statfs = async () => { events.push('statfs'); return { bsize: 4096, blocks: 100, bavail: 50 }; };
    await readSnapshot({ dataDir: '/tmp', fsApi, env, statfs, limits });
    assert.deepEqual(calls.filter((call) => call.name === 'observeSnapshot').map((call) => Object.keys(call.args[0]).sort()), [['env', 'fsApi']]);
    const [{ fsApi: givenFs, env: givenEnv }] = calls.find((call) => call.name === 'observeSnapshot').args;
    assert.equal(givenFs, fsApi);
    assert.equal(givenEnv, env);
    assert.equal(calls.find((call) => call.name === 'decorateSnapshot').args[0].observation, observation);
    const at = (event) => events.indexOf(event);
    assert.ok(at('statfs') >= 0 && at('statfs') < at('observeSnapshot'), 'the disk read comes first');
    assert.ok(at('observeSnapshot') >= 0 && at('observeSnapshot') < at('read:/proc/meminfo'), 'observeSnapshot comes before the memory read');
    assert.ok(at('observeSnapshot') < at('capCores'), 'observeSnapshot comes before the core count is capped');
    assert.equal(events.filter((event) => event === 'observeSnapshot').length, 1);
    assert.equal(events.at(-1), 'decorateSnapshot', 'decorateSnapshot comes last');
    // Under the identity default nothing is observed and the decorator receives null.
    const plain = recording();
    await readSnapshot({ dataDir: '/tmp', fsApi: cpuFs(), env, statfs, limits: plain.limits });
    assert.equal(plain.calls.find((call) => call.name === 'decorateSnapshot').args[0].observation, null);
});

test('observeSnapshot: a memory read that throws still leaves the observation made, and decorateSnapshot is not called', async () => {
    const events = [];
    const base = cpuFs();
    const fsApi = {
        readFileSync(file, ...rest) {
            if (file === '/proc/meminfo') { events.push('meminfo-throws'); throw Object.assign(new Error('meminfo unreadable'), { code: 'EIO' }); }
            return base.readFileSync(file, ...rest);
        },
        readdirSync(...args) { return base.readdirSync(...args); },
    };
    const { limits, calls } = recording({ observeSnapshot: () => { events.push('observeSnapshot'); return { observed: true }; } });
    await assert.rejects(readSnapshot({ dataDir: '/tmp', fsApi, env: {}, limits, statfs: async () => ({ bsize: 1, blocks: 1, bavail: 1 }) }), /meminfo unreadable/);
    assert.deepEqual(events, ['observeSnapshot', 'meminfo-throws']);
    assert.equal(calls.filter((call) => call.name === 'observeSnapshot').length, 1);
    assert.equal(calls.filter((call) => call.name === 'decorateSnapshot').length, 0);
});

// ---------------------------------------------------------------- admit, refuseProfile, effective

const model = Object.freeze({ id: 'm', displayName: 'Model M', memory: Object.freeze({ layers: 24, kvBytesPerToken: 12288 }) });
const source = Object.freeze({ type: 'gguf', size: 400 * MIB });
const dedicatedGpu = Object.freeze({ available: true, backend: 'cuda', memoryModel: 'dedicated', name: 'GPU', totalBytes: 12 * GIB, freeBytes: 11 * GIB });
const vulkanGpu = Object.freeze({ available: true, backend: 'vulkan', driver: 'radv', memoryModel: 'dedicated', name: 'GPU', totalBytes: 12 * GIB, freeBytes: 11 * GIB });
const hostMemory = Object.freeze({ totalBytes: 32 * GIB, availableBytes: 30 * GIB });
const snapshotOf = (gpu) => ({ gpu, memory: { ...hostMemory }, disk: { freeBytes: 1000 * GIB } });

/** A runner whose policies record the input they receive and answer ok. */
function stubRunner(extra = {}) {
    const seen = {};
    const policy = (name) => (input) => { seen[name] = input; return admissionResult('ok', null, {}); };
    return { seen, runner: { id: 'stub', displayName: 'Stub', supported: true, admit: policy('admit'), admitUnified: policy('admitUnified'),
        admitCpu: policy('admitCpu'), admitVulkan: policy('admitVulkan'), ...extra } };
}

test('refuseProfile and admit: the engine asks in order, before any policy, with the documented inputs, and a refusal ends the admission', () => {
    const { runner, seen } = stubRunner();
    const snapshot = snapshotOf(null);
    const env = { MARK: '1' };
    const probe = recording();
    const answer = admit({ runner, model, source, params: {}, snapshot, profile: 'cpu', limits: probe.limits, env });
    assert.equal(answer.status, 'ok');
    assert.deepEqual(probe.calls.map((call) => `${call.name}:${call.args[0].stage}`), ['refuseProfile:admission', 'admit:before-policy']);
    const [{ stage, profile, snapshot: sentTo, refuse }] = probe.named('refuseProfile')[0].args;
    assert.deepEqual([stage, profile, sentTo], ['admission', 'cpu', snapshot]);
    // The refuse builder makes the admission result with the constraint in the estimate and no warnings.
    const built = refuse('incompatible', 'C3', 'why');
    assert.deepEqual(built, { status: 'incompatible', reason: 'why', estimate: { constraint: 'C3', isEstimate: true }, warnings: [] });
    assert.ok(Object.isFrozen(built));
    const gate = probe.named('admit')[0].args[0];
    assert.equal(gate.profile, 'cpu');
    assert.equal(gate.snapshot, snapshot);
    assert.equal(gate.env, env);
    const coded = gate.result('insufficient-now', 'later', { a: 1 }, ['w'], 'a_code');
    assert.deepEqual(coded, { status: 'insufficient-now', reason: 'later', estimate: { a: 1, isEstimate: true }, warnings: ['w'], reasonCode: 'a_code' });
    // The CPU policy gets the snapshot and the limits to size against.
    assert.equal(seen.admitCpu.snapshot, snapshot);
    assert.equal(seen.admitCpu.limits, probe.limits);
    assert.equal(seen.admitCpu.memory, snapshot.memory);

    // A refusal from the first hook ends it: neither the second hook nor a policy runs.
    const refused = recording({ refuseProfile: ({ refuse }) => refuse('incompatible', 'C3', 'no APU here') });
    const first = admit({ runner, model, source, params: {}, snapshot, profile: 'apu', limits: refused.limits });
    assert.deepEqual([first.status, first.reason, first.estimate.constraint], ['incompatible', 'no APU here', 'C3']);
    assert.deepEqual(refused.calls.map((call) => call.name), ['refuseProfile']);
    // A refusal from the second ends it before the policy.
    const gated = recording({ admit: ({ stage, result }) => stage === 'before-policy' ? result('insufficient-now', 'cannot read', {}, [], 'a_code') : null });
    const second = admit({ runner, model, source, params: {}, snapshot, profile: 'cpu', limits: gated.limits });
    assert.deepEqual([second.status, second.reason, second.reasonCode], ['insufficient-now', 'cannot read', 'a_code']);
    assert.equal(Object.keys(seen).includes('admitCpu'), true);
    const before = seen.admitCpu;
    assert.equal(before.limits, probe.limits, 'the earlier call is unchanged');
});

test('refuseProfile and admit: nothing is asked of the limits before the runner and source checks', () => {
    const { runner } = stubRunner();
    const probe = recording();
    assert.equal(admit({ runner: { ...runner, supported: false }, model, source, params: {}, snapshot: snapshotOf(null), profile: 'cpu', limits: probe.limits }).status, 'incompatible');
    assert.equal(admit({ runner, model, source: null, params: {}, snapshot: snapshotOf(null), profile: 'cpu', limits: probe.limits }).status, 'incompatible');
    assert.equal(probe.calls.length, 0);
});

test('admit stage vulkan: asked after the policy check and before the device match; effective memory and the limits reach the policy', () => {
    const { runner, seen } = stubRunner();
    const snapshot = snapshotOf(vulkanGpu);
    const capped = { totalBytes: 2 * GIB, availableBytes: 1 * GIB };
    const probe = recording({ effective: () => ({ memory: capped }) });
    const ok = admit({ runner, model, source, params: {}, snapshot, profile: 'vulkan', limits: probe.limits });
    assert.equal(ok.status, 'ok');
    assert.deepEqual(probe.calls.map((call) => `${call.name}:${call.args[0].stage ?? ''}`), ['refuseProfile:admission', 'admit:before-policy', 'admit:vulkan', 'effective:']);
    assert.equal(probe.named('effective')[0].args[0].snapshot, snapshot);
    assert.equal(seen.admitVulkan.memory, capped);
    assert.equal(seen.admitVulkan.snapshot, snapshot);
    assert.equal(seen.admitVulkan.limits, probe.limits);
    assert.equal(seen.admitVulkan.gpu, vulkanGpu);

    // A runner with no policy for the profile is refused before the hook is asked.
    const without = recording();
    const noPolicy = admit({ runner: { ...runner, admitVulkan: undefined }, model, source, params: {}, snapshot, profile: 'vulkan', limits: without.limits });
    assert.match(noPolicy.reason, /use llama.cpp \(Vulkan build\)/);
    assert.equal(without.named('admit').filter((call) => call.args[0].stage === 'vulkan').length, 0);
    // The hook's refusal wins over a device that no longer matches.
    const mismatched = snapshotOf({ ...vulkanGpu, driver: 'venus' });
    const refusing = recording({ admit: ({ stage, result }) => stage === 'vulkan' ? result('incompatible', 'shares are not for Vulkan', {}, [], 'share_code') : null });
    const refused = admit({ runner, model, source, params: {}, snapshot: mismatched, profile: 'vulkan', limits: refusing.limits });
    assert.deepEqual([refused.reason, refused.reasonCode], ['shares are not for Vulkan', 'share_code']);
    const unrefused = admit({ runner, model, source, params: {}, snapshot: mismatched, profile: 'vulkan', limits: IDENTITY_LIMITS, messages: PLOINKY_MESSAGES });
    assert.match(unrefused.reason, /restart local-llm/);
});

test('admit stage gpu: asked once the GPU is known usable and matches the profile; its refusal, or its share view of the GPU, is used', () => {
    const { runner, seen } = stubRunner();
    const snapshot = snapshotOf(dedicatedGpu);
    const env = {};
    const shareGpu = { ...dedicatedGpu, totalBytes: 4 * GIB, freeBytes: 3 * GIB };
    const effectiveMemory = { totalBytes: 8 * GIB, availableBytes: 7 * GIB };
    const probe = recording({ admit: ({ stage }) => stage === 'gpu' ? { share: { gpu: shareGpu } } : null, effective: () => ({ memory: effectiveMemory }) });
    assert.equal(admit({ runner, model, source, params: {}, snapshot, profile: 'dedicated', limits: probe.limits, env }).status, 'ok');
    const device = probe.named('admit').find((call) => call.args[0].stage === 'gpu').args[0];
    assert.deepEqual([device.profile, device.selected, device.gpu, device.snapshot, device.env], ['dedicated', 'dedicated', dedicatedGpu, snapshot, env]);
    assert.equal(typeof device.result, 'function');
    assert.equal(seen.admit.gpu, shareGpu);
    assert.equal(seen.admit.memory, effectiveMemory);
    assert.equal(seen.admit.limits, probe.limits);

    // Unified: the same hook with selected 'unified'; the unified policy sizes against the effective memory.
    const unifiedGpu = { ...dedicatedGpu, memoryModel: 'unified' };
    const unified = recording({ effective: () => ({ memory: effectiveMemory }) });
    assert.equal(admit({ runner, model, source, params: {}, snapshot: snapshotOf(unifiedGpu), profile: 'unified', limits: unified.limits }).status, 'ok');
    assert.equal(unified.named('admit').find((call) => call.args[0].stage === 'gpu').args[0].selected, 'unified');
    assert.equal(seen.admitUnified.memory, effectiveMemory);
    assert.equal(seen.admitUnified.gpu, unifiedGpu);

    // A refusal is returned as it is, and no policy runs for it.
    const before = { ...seen };
    const refusing = recording({ admit: ({ stage, result }) => stage === 'gpu' ? { refusal: result('incompatible', 'not for this GPU', {}, [], 'gpu_code') } : null });
    const refused = admit({ runner, model, source, params: {}, snapshot, profile: 'dedicated', limits: refusing.limits });
    assert.deepEqual([refused.status, refused.reason, refused.reasonCode], ['incompatible', 'not for this GPU', 'gpu_code']);
    assert.equal(seen.admit, before.admit);
    assert.equal(refusing.named('effective').length, 0);
    // Not asked for a GPU that is gone, or for a profile that no longer matches it.
    const gone = recording();
    admit({ runner, model, source, params: {}, snapshot: snapshotOf({ available: false, state: 'absent' }), profile: 'dedicated', limits: gone.limits });
    admit({ runner, model, source, params: {}, snapshot: snapshotOf(dedicatedGpu), profile: 'unified', limits: gone.limits });
    assert.equal(gone.named('admit').filter((call) => call.args[0].stage === 'gpu').length, 0);
});

test('R12: a runner that requires share qualification is refused under a share unless the bound qualification passes it', () => {
    const shareGpu = { ...dedicatedGpu, totalBytes: 4 * GIB };
    const share = { admit: ({ stage }) => stage === 'gpu' ? { share: { gpu: shareGpu } } : null };
    const snapshot = snapshotOf(dedicatedGpu);
    const gated = stubRunner({ requiresShareQualification: true });
    const plain = stubRunner();
    const runnerContext = { runnerLockEntry: { id: 'stub', version: '1' } };

    // No qualification bound: refused with neutral wording and a neutral code; a runner that does not require it proceeds.
    const unbound = recording(share);
    const refused = admit({ runner: gated.runner, model, source, params: {}, snapshot, profile: 'dedicated', limits: unbound.limits, runnerContext });
    assert.deepEqual([refused.status, refused.reasonCode], ['incompatible', 'share_unqualified']);
    assert.match(refused.reason, /Stub cannot be used under a GPU share here/);
    assert.doesNotMatch(refused.reason, /ploinky|mps|cuda_mps/i);
    assert.equal(gated.seen.admit, undefined);
    assert.equal(admit({ runner: plain.runner, model, source, params: {}, snapshot, profile: 'dedicated', limits: unbound.limits, runnerContext }).status, 'ok');
    assert.equal(plain.seen.admit.gpu, shareGpu);

    // Bound: called with the runner context, the runner and the GPU as read; its refusal is used, and so is its pass.
    const refusing = recording(share, { qualification: () => ({ qualified: false, code: 'not_qualified', fix: 'Use another runner.' }) });
    const no = admit({ runner: gated.runner, model, source, params: {}, snapshot, profile: 'dedicated', limits: refusing.limits, runnerContext });
    assert.deepEqual([no.status, no.reason, no.reasonCode], ['incompatible', 'Use another runner.', 'not_qualified']);
    const [context] = refusing.named('vllmQualification')[0].args;
    assert.equal(context.runnerLockEntry, runnerContext.runnerLockEntry);
    assert.equal(context.runner, gated.runner);
    assert.equal(context.gpu, dedicatedGpu, 'the GPU as read, not the share view');
    const passing = recording(share, { qualification: () => ({ qualified: true }) });
    assert.equal(admit({ runner: gated.runner, model, source, params: {}, snapshot, profile: 'dedicated', limits: passing.limits, runnerContext }).status, 'ok');
    assert.equal(gated.seen.admit.gpu, shareGpu);

    // Without a share, nothing is qualified, flagged runner or not.
    const noShare = recording({}, { qualification: () => assert.fail('no share applies') });
    assert.equal(admit({ runner: gated.runner, model, source, params: {}, snapshot, profile: 'dedicated', limits: noShare.limits, runnerContext }).status, 'ok');
    // The real vLLM adapter carries the flag, and the other adapters do not.
    assert.equal(RUNNERS.vllm.requiresShareQualification, true);
    assert.deepEqual(Object.values(RUNNERS).filter((runner) => runner.requiresShareQualification).map((runner) => runner.id), ['vllm']);
});

test('refuseProfile stage policy: the APU policy asks after its device-memory checks and returns the hook refusal with its estimate and warnings', () => {
    const apuGpu = { available: true, backend: 'vulkan', driver: 'radv', memoryModel: 'shared', name: 'AMD Radeon Graphics', totalBytes: 16 * GIB, freeBytes: 16 * GIB };
    const snapshot = snapshotOf(apuGpu);
    const input = (limits, extra = {}) => ({ model: { displayName: 'M', memory: { kvBytesPerToken: 1024 } }, source: { size: 400 * MIB },
        params: { ctxSize: 2048, parallel: 1 }, gpu: apuGpu, memory: snapshot.memory, snapshot, limits, disk: snapshot.disk, ...extra });
    const probe = recording({ refuseProfile: ({ stage, refuse }) => stage === 'policy' ? refuse('incompatible', 'C3', 'limited here') : null });
    const refused = admitApuLlamaServer(input(probe.limits), admissionResult);
    assert.deepEqual([refused.status, refused.reason, refused.estimate.constraint], ['incompatible', 'limited here', 'C3']);
    assert.ok(refused.warnings.length > 0 && refused.estimate.weightsBytes === 400 * MIB, 'the policy estimate and warnings are kept');
    assert.deepEqual(probe.calls.map((call) => call.name), ['memoryPool', 'refuseProfile']);
    const [context] = probe.named('refuseProfile')[0].args;
    assert.deepEqual([context.stage, context.profile, context.snapshot], ['policy', 'apu', snapshot]);
    // The checks before it come first: unreadable device memory is C2, whatever the hook would say.
    const first = admitApuLlamaServer(input(probe.limits, { gpu: { ...apuGpu, totalBytes: 0 } }), admissionResult);
    assert.equal(first.estimate.constraint, 'C2');
    // The default asks nothing of the machine.
    assert.equal(admitApuLlamaServer(input(IDENTITY_LIMITS), admissionResult).status, 'ok');
});

// ---------------------------------------------------------------- memoryPool

const OLLAMA = getRunner('ollama');
const OLLAMA_MODEL = Object.freeze({ id: 'qwen', displayName: 'Qwen 0.5B', seed: true, contextLength: 32768, memory: Object.freeze({ layers: 24, kvBytesPerToken: 12288 }) });

function onCpu(limits, { memory = { totalBytes: 8 * GIB, availableBytes: 7 * GIB }, model = OLLAMA_MODEL, gpu = { available: false, state: 'absent' },
    decision = { cause: 'absent', reason: 'No GPU.' } } = {}) {
    const snapshot = { gpu, memory, disk: { freeBytes: 100 * GIB } };
    const result = admit({ runner: OLLAMA, model, source: { type: 'ollama', tag: 't', size: 400 * MIB },
        params: OLLAMA.normalizeParams({}, { model, profile: 'cpu' }), snapshot, profile: 'cpu',
        decision, limits });
    return { result, snapshot };
}

test('memoryPool: the CPU policy asks with the memory, the snapshot and a byte formatter, sizes against the answer and adds its warning last', () => {
    const pool = { totalBytes: 2 * GIB, availableBytes: GIB, capped: true, warning: 'A limit of two applies.' };
    const probe = recording({ memoryPool: pool });
    const { result, snapshot } = onCpu(probe.limits);
    const [memory, sent, helpers] = probe.named('memoryPool')[0].args;
    assert.equal(memory, snapshot.memory);
    assert.equal(sent, snapshot);
    assert.equal(helpers.amount(2 * GIB), '2.00 GiB');
    assert.equal(helpers.amount(512 * MIB), '512 MiB');
    assert.equal(result.estimate.poolBytes, 2 * GIB);
    assert.equal(result.warnings.at(-1), 'A limit of two applies.');
    // The physical pool when the hook answers null, and no warning for a pool that is not capped.
    assert.equal(onCpu(recording({ memoryPool: null }).limits).result.estimate.poolBytes, 8 * GIB);
    const uncapped = onCpu(recording({ memoryPool: { ...pool, capped: false } }).limits).result;
    assert.equal(uncapped.estimate.poolBytes, 2 * GIB);
    assert.ok(!uncapped.warnings.includes('A limit of two applies.'));
    // The pool warning sits between the user-sizing warning and the restart hint.
    const added = { ...OLLAMA_MODEL, seed: false, memory: { ...OLLAMA_MODEL.memory } };
    const recovered = onCpu(recording({ memoryPool: pool }).limits, { model: added, gpu: { ...dedicatedGpu },
        decision: { cause: 'unreadable-timeout', reason: 'The GPU could not be read.', gpuRecovered: true } }).result;
    const tail = recovered.warnings.slice(-3);
    assert.match(tail[0], /^Sized with memory\.kvBytesPerToken from the model entry added at run time\./);
    assert.equal(tail[1], 'A limit of two applies.');
    assert.match(tail[2], /^GPU access is available now; restart to select it\./);
    assert.equal(recovered.warnings.filter((warning) => /^Sized with /.test(warning)).length, 1);
    // Memory that cannot be read is the engine's own refusal; the hook is not asked.
    const unread = recording({ memoryPool: pool });
    const refused = onCpu(unread.limits, { memory: {} }).result;
    assert.equal(refused.status, 'incompatible');
    assert.match(refused.reason, /cannot be read/);
    assert.equal(unread.calls.filter((call) => call.name === 'memoryPool').length, 0);
});

// A limit that is a hard ceiling (a container's memory) returns a capped view with `physicalTotalBytes` and
// `physicalAvailableBytes` beside it, as the denominator of a vLLM share. Those members do not move the host reserve or the floor:
// they stay inside the capped figures, as they were before `reserveTotalBytes` and `reserveAvailableBytes` existed.
test('effective and memoryPool: a view that carries physicalTotalBytes and physicalAvailableBytes keeps its reserve and floor inside the capped figures', () => {
    const seedModel = loadSeedCatalog(SEED_CATALOG_FILE).find((entry) => entry.id === 'qwen2.5-0.5b-instruct-q4_k_m');
    const llama = getRunner('llama.cpp');
    const gb10 = { available: true, name: 'NVIDIA GB10', memoryModel: 'unified', totalBytes: null, usedBytes: null, freeBytes: null, processes: [] };
    const unifiedVerdict = (memory, view) => {
        const snapshot = { gpu: gb10, memory, disk: { freeBytes: 400 * GIB } };
        const limits = defineLimits({ effective: () => ({ memory: view }) });
        return admit({ runner: llama, model: seedModel, source: seedModel.sources.gguf, snapshot, profile: 'unified', limits,
            params: llama.normalizeParams({}, { model: seedModel, profile: 'unified' }) });
    };
    const host = { totalBytes: 128 * GIB, availableBytes: 120 * GIB };
    const need = unifiedVerdict(host, host).estimate.unifiedBytes;
    const view = (total, available, extra = { physicalTotalBytes: host.totalBytes, physicalAvailableBytes: host.availableBytes }) => ({ totalBytes: total, availableBytes: available, ...extra });
    // The reserve: the need fits exactly when it is the capped total less the 16 GiB reserve, whatever the host holds.
    const atEdge = unifiedVerdict(host, view(Math.ceil(need) + UNIFIED.hostReserveBytes, 100 * GIB));
    assert.equal(atEdge.status, 'ok', atEdge.reason);
    const pastEdge = unifiedVerdict(host, view(Math.ceil(need) + UNIFIED.hostReserveBytes - 1, 100 * GIB));
    assert.equal(pastEdge.status, 'incompatible', pastEdge.reason);
    assert.match(pastEdge.reason, /which must keep 16\.0 GiB for the host/);
    // The floor: the need fits now exactly when the capped available memory less the 8 GiB floor holds it.
    const noEdge = unifiedVerdict(host, view(100 * GIB, Math.ceil(need) + UNIFIED.floorBytes));
    assert.equal(noEdge.status, 'ok', noEdge.reason);
    const pastFloor = unifiedVerdict(host, view(100 * GIB, Math.ceil(need) + UNIFIED.floorBytes - 1));
    assert.equal(pastFloor.status, 'insufficient-now', pastFloor.reason);
    assert.equal(pastFloor.estimate.floorBytes, UNIFIED.floorBytes);
    assert.equal(pastFloor.estimate.hostReserveBytes, UNIFIED.hostReserveBytes);
    // Whole verdicts are those of a view without the members.
    const bare = view(Math.ceil(need) + UNIFIED.hostReserveBytes - 1, 100 * GIB, {});
    assert.deepEqual(unifiedVerdict(host, view(bare.totalBytes, bare.availableBytes)), unifiedVerdict(host, bare));

    // The CPU pool: its floor and reserve are those of the capped total.
    const pool = { totalBytes: 4 * GIB, availableBytes: 3 * GIB, capped: true, warning: 'A limit of four applies.' };
    const members = { physicalTotalBytes: 64 * GIB, physicalAvailableBytes: 60 * GIB };
    const bareCpu = onCpu(recording({ memoryPool: pool }).limits).result;
    const withMembers = onCpu(recording({ memoryPool: { ...pool, ...members } }).limits).result;
    assert.equal(withMembers.estimate.floorBytes, cpuFloorBytes(4 * GIB));
    assert.equal(withMembers.estimate.hostReserveBytes, cpuHostReserveBytes(4 * GIB));
    assert.equal(withMembers.estimate.floorBytes, 512 * MIB);
    assert.equal(withMembers.estimate.hostReserveBytes, 1.5 * GIB);
    assert.deepEqual(withMembers, bareCpu);
});

// ---------------------------------------------------------------- refuseResourceRequest

test('refuseResourceRequest: the Ollama policy asks after its GPU checks with the sized GPU and estimate, and returns the hook refusal', () => {
    const params = OLLAMA.normalizeParams({ numGpu: 12 }, { model: OLLAMA_MODEL, profile: 'dedicated' });
    const gpu = { ...dedicatedGpu };
    const input = (limits) => ({ model: OLLAMA_MODEL, source: { type: 'ollama', tag: 't', size: 400 * MIB }, params, gpu, memory: hostMemory, disk: { freeBytes: 100 * GIB }, limits });
    const probe = recording({ refuseResourceRequest: ({ refuse }) => refuse('insufficient-now', 'the share is short') });
    const refused = admitOllama(input(probe.limits));
    assert.deepEqual([refused.status, refused.reason], ['insufficient-now', 'the share is short']);
    assert.equal(refused.estimate.ramBytes > 0 && refused.estimate.gpuBytes > 0, true, 'the estimate with RAM is kept');
    const [context] = probe.named('refuseResourceRequest')[0].args;
    assert.equal(context.policy, 'ollama');
    assert.equal(context.gpu, gpu);
    assert.equal(context.estimate.gpuBytes, refused.estimate.gpuBytes);
    assert.equal(context.marginBytes, 256 * MIB);
    assert.equal(context.gib(GIB), '1.0 GiB');
    assert.equal(admitOllama(input(IDENTITY_LIMITS)).status, 'ok');
    // A GPU that is too small is refused before the hook is asked.
    const early = recording({ refuseResourceRequest: () => assert.fail('asked too early') });
    assert.equal(admitOllama({ ...input(early.limits), gpu: { ...gpu, freeBytes: GIB / 2 } }).status, 'insufficient-now');
});

test('refuseResourceRequest: the vLLM policy asks with the request and the denominator it computed, and returns the hook refusal', () => {
    const gpu = { available: true, totalBytes: 12 * GIB, freeBytes: 11 * GIB, deviceTotalBytes: 24 * GIB };
    const params = { maxModelLen: 2048, cpuOffloadGb: 0, gpuMemoryUtilization: 0.4, kvCacheDtype: null };
    const input = (limits) => ({ model: { id: 'v', displayName: 'V', memory: { kvBytesPerToken: 1024 } }, source: { size: GIB }, params, gpu, memory: hostMemory, disk: { freeBytes: 100 * GIB }, limits });
    const probe = recording({ refuseResourceRequest: ({ refuse }) => refuse('incompatible', 'over the share') });
    const refused = admitVllm(input(probe.limits));
    assert.deepEqual([refused.status, refused.reason], ['incompatible', 'over the share']);
    const [context] = probe.named('refuseResourceRequest')[0].args;
    assert.equal(context.policy, 'vllm');
    assert.equal(context.gpu, gpu);
    assert.equal(context.params, params);
    assert.equal(context.gpuMemoryUtilization, 0.4);
    assert.equal(context.denominator, 24 * GIB, 'the device total when the GPU has one');
    assert.equal(context.estimate.weightsBytes, GIB);
    assert.equal(context.gib(3 * GIB), '3.0 GiB');
    assert.equal(refused.estimate.weightsBytes, GIB);
    assert.equal(admitVllm(input(IDENTITY_LIMITS)).status, 'ok');
});

// ---------------------------------------------------------------- the controller: overview and runnerEnvironment

function cpuController(t, limits, { env = {}, startEnv = [], startArgs = [] } = {}) {
    const dataDir = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), 'limits-hooks-'));
    const weight = path.join(dataDir, 'weight.gguf');
    fs.writeFileSync(weight, 'weights');
    const memory = { totalBytes: 8 * GIB, availableBytes: 7 * GIB };
    const seed = loadSeedCatalog(SEED_CATALOG_FILE).filter((entry) => entry.id === 'qwen2.5-0.5b-instruct-q4_k_m').map((entry) => validateModel(entry, { seed: true }));
    const snapshot = { gpu: { available: false, state: 'absent' }, memory, disk: { freeBytes: 1e12 } };
    const controller = createController({ dataDir, seedCatalog: seed, sharedModelsRoot: null, imageContract: {}, limits,
        env: { PATH: '/usr/bin', ...env }, fileExists: () => true, installer: { installable: () => false }, detectRunner: () => ({ installed: true }),
        snapshot: async () => snapshot, readMemory: () => memory, readPressure: () => 0,
        inspect: async () => ({ state: 'complete', bytes: 7, total: 7 }),
        download: async () => ({ status: 'complete', path: weight, bytesTransferred: 0 }),
        dropCache: () => true, pollMs: 2, stopGraceMs: 20,
        startRunner(options) {
            startEnv.push(options.env);
            startArgs.push(options.args);
            let resolve;
            const handle = { pid: 9000, running: true, exited: new Promise((done) => { resolve = done; }),
                async stop() { handle.running = false; resolve({ code: 0, signal: 'SIGTERM' }); }, async kill() { return handle.stop(); } };
            return handle;
        },
        fetchImpl: async () => ({ ok: true, status: 200, json: async () => ({}) }) });
    t.after(async () => { await controller.drain(); fs.rmSync(dataDir, { recursive: true, force: true }); });
    return { controller, snapshot, env: { PATH: '/usr/bin', ...env }, seed };
}

async function until(predicate) {
    const deadline = performance.now() + 3000;
    while (performance.now() < deadline) { if (await predicate()) return; await new Promise((resolve) => setTimeout(resolve, 2)); }
    assert.fail('not reached');
}

test('overview: the controller publishes the hook answer as the limits budget, with the snapshot and env, and nothing by default', async (t) => {
    const budget = { cpus: 1.5, source: 'fake' };
    const env = { MARK: '1' };
    const probe = recording({ overview: budget });
    const { controller, snapshot } = cpuController(t, probe.limits, { env });
    const view = await controller.overview();
    assert.equal(view.profile, 'cpu');
    assert.equal(view.limits.budget, budget);
    assert.ok(view.limits.floorBytes > 0, 'the pool figures are published beside it');
    const [context] = probe.named('overview')[0].args;
    assert.equal(context.snapshot, snapshot);
    assert.equal(context.env.MARK, '1');
    assert.equal(probe.named('memoryPool')[0].args[1], snapshot, 'the pool of the floor is the hook pool');
    const plain = await cpuController(t, undefined).controller.overview();
    assert.equal('budget' in plain.limits, false);
    // A budget alone is published for a profile with no pool figures.
    assert.deepEqual(Object.keys(plain.limits).sort(), ['availableBytes', 'floorBytes', 'hostReserveBytes', 'poolBytes']);
});

test('runnerEnvironment: the hook gets the profile, the controller env and a copy of the adapter environment, and its answer is what the runner is started with', async (t) => {
    const started = [];
    const probe = recording({ runnerEnvironment: ({ adapterEnvironment }) => {
        const { CUDA_VISIBLE_DEVICES, ...rest } = adapterEnvironment;
        return { ...rest, ADDED_BY_LIMITS: '1' };
    } });
    const { controller, seed } = cpuController(t, probe.limits, { startEnv: started });
    await controller.run({ modelId: seed[0].id, runnerId: 'llama.cpp', requestId: 'limits-env-0001', params: {} });
    await until(() => started.length === 1);
    const [context] = probe.named('runnerEnvironment')[0].args;
    assert.equal(context.profile, 'cpu');
    assert.equal(context.env.PATH, '/usr/bin');
    assert.deepEqual(context.adapterEnvironment, { CUDA_VISIBLE_DEVICES: '' }, 'what the CPU adapter asked for');
    assert.equal(started[0].ADDED_BY_LIMITS, '1');
    assert.equal('CUDA_VISIBLE_DEVICES' in started[0], false, 'the hook removed what it manages');
    assert.deepEqual(Object.keys(started[0]).slice(0, 4), ['PATH', 'HOME', 'LANG', 'CUDA_CACHE_PATH']);
    // The default leaves the adapter's environment as it asked.
    const plainStarted = [];
    const plain = cpuController(t, undefined, { startEnv: plainStarted });
    await plain.controller.run({ modelId: plain.seed[0].id, runnerId: 'llama.cpp', requestId: 'limits-env-0002', params: {} });
    await until(() => plainStarted.length === 1);
    assert.equal(plainStarted[0].CUDA_VISIBLE_DEVICES, '');
    assert.equal('ADDED_BY_LIMITS' in plainStarted[0], false);
});

test('capCores: a runner the controller builds defaults its threads from the count the hook caps', async (t) => {
    const probe = recording({ capCores: () => 2 });
    const args = [];
    const { controller, seed } = cpuController(t, probe.limits, { startArgs: args });
    await controller.run({ modelId: seed[0].id, runnerId: 'llama.cpp', requestId: 'limits-cap-0001', params: {} });
    await until(() => args.length === 1);
    assert.ok(probe.named('capCores').length >= 1);
    const threadsFor = (limits) => String(performanceCoreCount({ limits }) ?? cpuDefaultThreads(physicalCoreCount({ limits })));
    const expected = threadsFor(recording({ capCores: () => 2 }).limits);
    assert.equal(args[0][args[0].indexOf('--threads') + 1], expected, 'the threads of the capped 2 cores');
    const plainArgs = [];
    const plain = cpuController(t, undefined, { startArgs: plainArgs });
    await plain.controller.run({ modelId: plain.seed[0].id, runnerId: 'llama.cpp', requestId: 'limits-cap-0002', params: {} });
    await until(() => plainArgs.length === 1);
    assert.equal(plainArgs[0][plainArgs[0].indexOf('--threads') + 1], threadsFor(undefined), 'the threads of the physical cores');
});

test('runnerEnvironment: a throw from the hook refuses the start', async (t) => {
    const probe = recording({ runnerEnvironment: () => { throw Object.assign(new Error('the share is incomplete'), { code: 'share_invalid' }); } });
    const started = [];
    const { controller, seed } = cpuController(t, probe.limits, { startEnv: started });
    await controller.run({ modelId: seed[0].id, runnerId: 'llama.cpp', requestId: 'limits-env-0003', params: {} });
    await until(() => controller.state.deployment?.phase === 'error');
    assert.match(controller.state.deployment.error, /the share is incomplete/);
    assert.equal(started.length, 0);
});

test('the controller hands its limits to the snapshot reader and the hardware readers', async (t) => {
    const probe = recording();
    const readers = [];
    const { controller } = cpuController(t, probe.limits);
    assert.ok(controller);
    const dataDir = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), 'limits-reader-'));
    t.after(() => fs.rmSync(dataDir, { recursive: true, force: true }));
    const own = createController({ dataDir, limits: probe.limits, imageContract: {}, readHardwareSnapshot: async (options) => { readers.push(options); return { gpu: { available: false, state: 'absent' }, memory: { totalBytes: 8 * GIB, availableBytes: 7 * GIB }, disk: { freeBytes: 1e12 } }; } });
    t.after(() => own.drain());
    await own.overview();
    assert.equal(readers[0].limits, probe.limits);
});

// ---------------------------------------------------------------- the dedicated guard (cadence and headroom off the APU)

const DEDICATED_SNAPSHOT = Object.freeze({
    gpu: { available: true, name: 'Test GPU', totalBytes: 6144 * MIB, usedBytes: 144 * MIB, freeBytes: 6000 * MIB, processes: [] },
    memory: { totalBytes: 31 * GIB, availableBytes: 24 * GIB },
    disk: { freeBytes: 300 * GIB, totalBytes: 500 * GIB },
});

// A GGUF runner whose admission records a RAM floor, so the dedicated profile starts the memory guard.
const FLOORED_RUNNER = Object.freeze({
    id: 'floored', displayName: 'Floored runner', weightFormat: 'gguf', pinnedVersion: 'v1', supported: true, executable: '/opt/fake/server',
    port: 18999, apiKey: true,
    paramSchema: Object.freeze({ type: 'object', additionalProperties: false, properties: {} }),
    basicParams: Object.freeze([]),
    normalizeParams: () => ({}),
    describeContext: () => ({ totalContext: 1, perRequestContext: 1, parallel: 1, kvUnified: false }),
    detect: () => ({ installed: true, version: 'v1', reason: null }),
    async start(ctx) {
        const process = ctx.launch({ command: '/opt/fake/server', args: [], env: {} });
        await ctx.waitForHttp(`http://127.0.0.1:${ctx.port}/ready`, { process });
        return {};
    },
    chatModel: (deployment) => deployment.modelId,
    admit: () => admissionResult('ok', null, { gpuBytes: GIB, ramFloorBytes: 4 * GIB, basis: 'fake policy' }),
    parseReport: () => ({}),
});

// The same runner with no RAM floor in its admission: the dedicated profile has no guard for it unless the limits ask for one.
const FLOORLESS_RUNNER = Object.freeze({
    ...FLOORED_RUNNER, id: 'floorless', displayName: 'Floorless runner',
    admit: () => admissionResult('ok', null, { gpuBytes: GIB, basis: 'fake policy' }),
});

function dedicatedGuard(t, limits, runner = FLOORED_RUNNER) {
    const dataDir = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), 'limits-guard-'));
    t.after(() => fs.rmSync(dataDir, { recursive: true, force: true }));
    const reads = { count: 0, availableBytes: 24 * GIB };
    const controller = createController({
        dataDir, env: { PATH: '/usr/bin' }, seedCatalog: loadSeedCatalog(SEED_CATALOG_FILE), stateStore: createStateStore({ dataDir }), limits,
        runners: { [runner.id]: runner }, snapshot: async () => structuredClone(DEDICATED_SNAPSHOT),
        download: async ({ artifact }) => ({ status: 'complete', path: `/data/models/gguf/${artifact.file}`, bytesTransferred: 0 }),
        inspect: async () => ({ state: 'complete', bytes: 1 }),
        startRunner({ command, args, env }) {
            let running = true; let resolveExit;
            const handle = { pid: 7000, command, args, env, exited: new Promise((resolve) => { resolveExit = resolve; }),
                get running() { return running; },
                async stop() { running = false; resolveExit({ code: 0, signal: 'SIGTERM', error: null }); return handle.exited; },
                async kill() { return handle.stop(); } };
            return handle;
        },
        fetchImpl: async () => ({ ok: true, status: 200, json: async () => ({}) }),
        detectRunner: (runner) => ({ installed: runner.supported, version: runner.pinnedVersion, reason: null }),
        pollMs: 2, stopGraceMs: 20,
        // The dedicated guard samples this slowly unless the limits ask to be watched; unifiedGuardMs is the short interval.
        readMemory: () => { reads.count += 1; return { totalBytes: 31 * GIB, availableBytes: reads.availableBytes }; },
        memoryGuardLoadMs: 60_000, memoryGuardReadyMs: 60_000, unifiedGuardMs: 5,
    });
    t.after(() => controller.drain());
    const run = (requestId) => controller.run({ modelId: 'gpt-oss-20b', runnerId: runner.id, requestId, params: {} });
    return { controller, reads, run };
}

test('guard hooks: a dedicated guard moves to the short interval only when the first sample is watched', async (t) => {
    const watched = dedicatedGuard(t, recording({ guardSample: { watched: true } }).limits);
    await watched.run('limits-dedicated-0001'); await until(() => watched.controller.state.deployment?.phase === 'ready');
    await until(() => watched.reads.count >= 5);
    const quiet = dedicatedGuard(t, recording({ guardSample: { watched: false } }).limits);
    await quiet.run('limits-dedicated-0002'); await until(() => quiet.controller.state.deployment?.phase === 'ready');
    await new Promise((resolve) => setTimeout(resolve, 150));
    assert.equal(quiet.reads.count, 0, 'the slow interval has not elapsed');
    const plain = dedicatedGuard(t, undefined);
    await plain.run('limits-dedicated-0003'); await until(() => plain.controller.state.deployment?.phase === 'ready');
    await new Promise((resolve) => setTimeout(resolve, 150));
    assert.equal(plain.reads.count, 0, 'the identity default keeps the slow interval');
});

test('guard hooks: the headroom answer is compared with the floor of a dedicated guard, which stops it without a kill', async (t) => {
    let cap = null;
    const probe = recording({ guardSample: { watched: true }, guardBreach: (sample, context) => context.check === 'headroom' ? { available: cap ?? context.available } : null });
    const h = dedicatedGuard(t, probe.limits);
    await h.run('limits-dedicated-0004'); await until(() => h.controller.state.deployment?.phase === 'ready');
    await until(() => probe.named('guardBreach').length >= 2);
    assert.deepEqual([...new Set(probe.named('guardBreach').map((call) => call.args[1].check))], ['headroom'], 'the watchable check is the APU guard\'s alone');
    const [sample, context] = probe.named('guardBreach')[0].args;
    assert.deepEqual(sample, { watched: true });
    assert.equal(context.available, 24 * GIB);
    assert.equal(context.memory.totalBytes, 31 * GIB);
    assert.equal(h.controller.state.deployment.phase, 'ready');
    cap = 0;
    await until(() => h.controller.state.deployment?.phase !== 'ready');
    assert.match(h.controller.state.deployment.error ?? '', /below the floor \(0\.0 GiB available, 4\.0 GiB required\)/);
});

// ---------------------------------------------------------------- the guard under limits that stop: a guard with no floor, and { stop }

const STOP = Object.freeze({ stop: 'stopped: the limits say so' });
const WATCHED = Object.freeze({ watched: true, standalone: true });
const quietFor = () => new Promise((resolve) => setTimeout(resolve, 150));

/**
 * A guardBreach answer that follows `script` (one answer per headroom call, then null), and counts the headroom calls. The guard
 * ticks from the launch, possibly before `ready`: the answers are null, and not counted, until the test sets `armed` after `ready`.
 */
function scripted(script) {
    const seen = { headroom: 0 };
    const scripts = { seen, armed: false };
    scripts.answer = (sample, context) => (context.check === 'headroom' && scripts.armed ? script[seen.headroom++] ?? null : null);
    return scripts;
}

test('guard hooks: a runner with no floor is guarded only when the first sample says watched and standalone', async (t) => {
    const identity = dedicatedGuard(t, undefined, FLOORLESS_RUNNER);
    await identity.run('limits-floorless-0001'); await until(() => identity.controller.state.deployment?.phase === 'ready');
    await quietFor();
    assert.equal(identity.reads.count, 0, 'the identity limits: no floor, no guard');
    // The shape the container limits give: watched under a finite memory limit, with no further member. No floor, so no guard, whatever
    // guardBreach would say, and the limits are sampled once and never again.
    const container = recording({ guardSample: { watched: true, finiteMemory: true }, guardBreach: { unreadable: 'stopped: the budget cannot be read' } });
    const unmarked = dedicatedGuard(t, container.limits, FLOORLESS_RUNNER);
    await unmarked.run('limits-floorless-0002'); await until(() => unmarked.controller.state.deployment?.phase === 'ready');
    await quietFor();
    assert.equal(unmarked.reads.count, 0, 'memory is not sampled');
    assert.equal(container.named('guardSample').length, 1, 'the limits are sampled once, to decide');
    assert.equal(container.named('guardBreach').length, 0);
    assert.equal(unmarked.controller.state.deployment.phase, 'ready');
    for (const first of [{ standalone: true }, { watched: false, standalone: true }, { watched: true, standalone: false }]) {
        const probe = recording({ guardSample: first });
        const h = dedicatedGuard(t, probe.limits, FLOORLESS_RUNNER);
        await h.run('limits-floorless-0003'); await until(() => h.controller.state.deployment?.phase === 'ready');
        await quietFor();
        assert.equal(h.reads.count, 0, JSON.stringify(first));
    }
    // Watched and standalone: the guard samples on the short interval and asks for headroom on every tick.
    const probe = recording({ guardSample: { ...WATCHED, usedBytes: 1 } });
    const h = dedicatedGuard(t, probe.limits, FLOORLESS_RUNNER);
    await h.run('limits-floorless-0004'); await until(() => h.controller.state.deployment?.phase === 'ready');
    await until(() => probe.named('guardBreach').length >= 3);
    assert.ok(h.reads.count >= 3);
    assert.deepEqual([...new Set(probe.named('guardBreach').map((call) => call.args[1].check))], ['headroom']);
    const [sample, context] = probe.named('guardBreach')[0].args;
    assert.deepEqual(sample, { ...WATCHED, usedBytes: 1 });
    assert.equal(context.available, 24 * GIB);
    assert.equal(context.memory.totalBytes, 31 * GIB);
    assert.equal(h.controller.state.deployment.phase, 'ready');
});

test('guard hooks: with no floor, { available } and { unreadable } answers are ignored; only two { stop } answers stop the runner', async (t) => {
    const seen = { headroom: 0 };
    const probe = recording({ guardSample: WATCHED, guardBreach: () => (seen.headroom++ % 3 === 0 ? { unreadable: 'stopped: unreadable' } : { available: 0 }) });
    const h = dedicatedGuard(t, probe.limits, FLOORLESS_RUNNER);
    await h.run('limits-floorless-0005'); await until(() => h.controller.state.deployment?.phase === 'ready');
    await until(() => seen.headroom >= 12);
    assert.equal(h.controller.state.deployment.phase, 'ready', 'no floor to compare { available: 0 } with, and no unreadable count');
});

for (const [name, runner] of [['a guard with a floor', FLOORED_RUNNER], ['a guard with no floor', FLOORLESS_RUNNER]]) {
    test(`guard hooks: two consecutive { stop } answers stop the runner with the message (${name})`, async (t) => {
        // stop, none, stop, an unreadable answer, stop, a figure, then two stops in a row.
        const script = [STOP, null, STOP, { unreadable: 'stopped: unreadable' }, STOP, { available: 24 * GIB }, STOP, STOP];
        const answers = scripted(script);
        const probe = recording({ guardSample: WATCHED, guardBreach: answers.answer });
        const h = dedicatedGuard(t, probe.limits, runner);
        await h.run('limits-stop-0001'); await until(() => h.controller.state.deployment?.phase === 'ready');
        answers.armed = true;
        await until(() => h.controller.state.deployment?.phase !== 'ready');
        assert.equal(h.controller.state.deployment.error, STOP.stop);
        assert.equal(answers.seen.headroom, script.length, 'it stopped on the second of the two, not before');
        await quietFor();
        assert.equal(answers.seen.headroom, script.length, 'and the guard ended with the runner');
    });

    test(`guard hooks: { stop } answers that never come twice in a row never stop the runner (${name})`, async (t) => {
        for (const between of [null, { unreadable: 'stopped: unreadable' }, { available: 24 * GIB }]) {
            const answers = scripted(Array.from({ length: 60 }, (_, index) => (index % 2 === 0 ? STOP : between)));
            const probe = recording({ guardSample: WATCHED, guardBreach: answers.answer });
            const h = dedicatedGuard(t, probe.limits, runner);
            await h.run('limits-stop-0002'); await until(() => h.controller.state.deployment?.phase === 'ready');
            answers.armed = true;
            await until(() => answers.seen.headroom >= 40);
            assert.equal(h.controller.state.deployment.phase, 'ready', JSON.stringify(between));
        }
    });
}

test('guard hooks: a physical floor breach still stops a guard with a floor at the first reading, with the engine\'s own message, whatever the limits answer', async (t) => {
    const probe = recording({ guardSample: WATCHED, guardBreach: () => null });
    const h = dedicatedGuard(t, probe.limits);
    await h.run('limits-stop-0003'); await until(() => h.controller.state.deployment?.phase === 'ready');
    h.reads.availableBytes = GIB;
    await until(() => h.controller.state.deployment?.phase !== 'ready');
    assert.equal(h.controller.state.deployment.error, 'stopped: host memory below the floor (1.0 GiB available, 4.0 GiB required)');
});

// The cpu and metal profiles, whose guards stop on a floor and on pressure: the stop answers count the same way on them.
function pooledGuard(t, { profile, limits, pressure, memory }) {
    const dataDir = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), 'limits-pooled-'));
    const seed = loadSeedCatalog(SEED_CATALOG_FILE);
    const tiny = seed.find((entry) => entry.id === 'qwen2.5-0.5b-instruct-q4_k_m');
    const metal = profile === 'metal';
    const handles = [];
    const machine = metal
        ? { gpu: { available: true, backend: 'metal', name: 'Apple M1', memoryModel: 'unified', deviceCapBytes: 12 * GIB },
            memory: { totalBytes: 16 * GIB, availableBytes: 9 * GIB, freeBytes: 2 * GIB, swapFreeBytes: GIB, cachedBytes: null, pressureLevel: 1 }, disk: { freeBytes: 200 * GIB, totalBytes: 500 * GIB }, cpus: 8, cores: 4 }
        : { gpu: { available: false }, memory: { totalBytes: 16 * GIB, availableBytes: 9 * GIB, pressureLevel: 1 }, disk: { freeBytes: 200 * GIB, totalBytes: 500 * GIB }, cpus: 8, cores: 4 };
    const controller = createController({
        dataDir, ...(metal ? {} : { hostOverrides: { platform: linuxPlatform() } }), env: { PATH: '/usr/bin' }, seedCatalog: seed, imageContract: null, hostArch: metal ? 'arm64' : 'x64',
        fileExists: () => true, sharedModelsRoot: null, profile, limits, detectRunner: () => ({ installed: true, version: 'b11295', reason: null }),
        snapshot: async () => structuredClone(machine), inspect: async () => ({ state: 'absent', bytes: 0 }),
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
        readMemory: () => memory.current, readPressure: () => pressure.current, metalGuardMs: 3, unifiedGuardMs: 3,
    });
    t.after(async () => { await controller.drain(); fs.rmSync(dataDir, { recursive: true, force: true }); });
    const run = async (requestId) => {
        await controller.run({ modelId: tiny.id, runnerId: 'llama.cpp', requestId });
        await until(() => controller.state.deployment?.phase === 'ready' || controller.state.deployment?.phase === 'error');
        assert.equal(controller.state.deployment.phase, 'ready', JSON.stringify(controller.state.deployment));
    };
    return { controller, handles, run };
}

for (const profile of ['cpu', 'metal']) {
    const normal = () => ({ current: profile === 'metal' ? 1 : 0 });
    const roomy = () => ({ current: { totalBytes: 16 * GIB, availableBytes: 9 * GIB } });

    test(`guard hooks: on the ${profile} profile two consecutive { stop } answers stop the runner under normal pressure and plenty of memory, and a stop and a null do not`, async (t) => {
        // Alternating answers: the runner stays up through many ticks.
        const alternating = scripted(Array.from({ length: 60 }, (_, index) => (index % 2 === 0 ? STOP : null)));
        const steady = pooledGuard(t, { profile, limits: recording({ guardSample: WATCHED, guardBreach: alternating.answer }).limits, pressure: normal(), memory: roomy() });
        await steady.run('limits-pooled-0001');
        alternating.armed = true;
        await until(() => alternating.seen.headroom >= 40);
        assert.equal(steady.controller.state.deployment.phase, 'ready');
        assert.equal(steady.handles[0].running, true);
        // Two in a row: stopped, with the limits' message, and the process group killed.
        const twice = scripted([null, STOP, null, STOP, STOP]);
        const h = pooledGuard(t, { profile, limits: recording({ guardSample: WATCHED, guardBreach: twice.answer }).limits, pressure: normal(), memory: roomy() });
        await h.run('limits-pooled-0002');
        twice.armed = true;
        await until(() => h.controller.state.deployment?.phase === 'error');
        assert.equal(h.controller.state.deployment.error, STOP.stop);
        assert.equal(twice.seen.headroom, 5);
        assert.equal(h.handles[0].killed, true);
    });

    test(`guard hooks: on the ${profile} profile the physical floor stop is the engine's own, whatever the limits answer`, async (t) => {
        const memory = roomy();
        const pressure = { current: profile === 'metal' ? 2 : 0 };
        const quiet = pooledGuard(t, { profile, limits: recording({ guardSample: WATCHED, guardBreach: () => null }).limits, pressure, memory });
        await quiet.run('limits-pooled-0003');
        memory.current = { totalBytes: 16 * GIB, availableBytes: cpuFloorBytes(16 * GIB) - 1 };
        await until(() => quiet.controller.state.deployment?.phase === 'error');
        // The Linux cpu guard stops on one reading below the floor; Metal on elevated pressure with the memory below it, three times in a row.
        assert.match(quiet.controller.state.deployment.error, profile === 'metal' ? /macOS memory pressure \(level 2\) for 3 samples in a row/ : /^stopped: host memory below the floor/);
        assert.equal(quiet.handles[0].killed, true);
    });
}

// ---------------------------------------------------------------- capThreads and capParams

test('capThreads and capParams: the identity returns its input unchanged', () => {
    const params = Object.freeze({ ctxSize: 4096, parallel: 1, threads: null });
    assert.equal(IDENTITY_LIMITS.capThreads(14, { profile: 'cpu' }), 14);
    assert.equal(IDENTITY_LIMITS.capParams({ runnerId: 'llama.cpp', profile: 'cpu', params, explicit: Object.freeze(['ctxSize']), model }), params);
});

const metalHost = (performanceCores) => agentLikeHost({ platform: Object.freeze({ ...HOST.platform, performanceCores: () => performanceCores }) });

test('capThreads: the default --threads of the cpu, dedicated, unified and metal launches is the hook\'s answer, and it is not called for a threads value the caller set', () => {
    const profiles = ['cpu', 'dedicated', 'unified', 'metal'];
    const { limits, calls } = recording({ capThreads: (threads) => Math.max(1, threads - 1) });
    // The metal and unified defaults are the platform's performance cores; the others the physical cores (each by the engine's own function).
    const runners = createRunners({ host: metalHost(10), messages: PLOINKY_MESSAGES, limits });
    const plain = createRunners({ host: metalHost(10), messages: PLOINKY_MESSAGES });
    for (const profile of profiles) {
        const before = Number(threadsOf(launchOf(plain['llama.cpp'], profile)));
        const after = Number(threadsOf(launchOf(runners['llama.cpp'], profile)));
        assert.ok(before > 1, `${profile}: ${before}`);
        assert.equal(after, before - 1, `${profile}: the hook lowered the default`);
    }
    assert.equal(threadsOf(launchOf(plain['llama.cpp'], 'metal')), '10', 'the Metal default is the performance cores of the platform');
    assert.deepEqual(calls.filter((call) => call.name === 'capThreads').map((call) => call.args[1]), profiles.map((profile) => ({ profile })));
    assert.equal(calls.filter((call) => call.name === 'capThreads').at(-1).args[0], 10, 'the hook gets the resolved default, the performance cores on metal');
    // A threads parameter the caller set is the caller's: the hook is not asked.
    const before = calls.length;
    const explicit = runners['llama.cpp'].buildLaunch({ artifactPath: '/m/q.gguf', params: { threads: 7 }, port: 18080, apiKey: 'k'.repeat(43),
        model: { id: 'm', displayName: 'M' }, runnerDir: '/x/run/llama.cpp/b1', profile: 'metal' });
    assert.equal(threadsOf(explicit), '7');
    assert.equal(calls.length, before);
});

test('capThreads: LM Studio\'s default threads go through the hook with the dedicated profile, and a threads value the caller set does not', async (t) => {
    const loadedThreads = async (params) => {
        const { limits, calls } = recording({ capThreads: () => 3 });
        const runner = createRunners({ host: HOST, messages: PLOINKY_MESSAGES, limits }).lmstudio;
        const runDir = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), 'limits-lms-'));
        t.after(() => fs.rmSync(runDir, { recursive: true, force: true }));
        const loads = [];
        const ctx = { runnerDir: runDir, model: loadSeedCatalog(SEED_CATALOG_FILE).find((entry) => entry.id === 'gpt-oss-20b'), params, port: 18084,
            weights: { path: '/data/models/gpt-oss-20b-MXFP4.gguf' }, launch: () => ({ running: true }), waitForHttp: async () => {},
            async exec(step) { if (step.args[1] === 'load') loads.push(JSON.parse(step.args[2])); return step.args[0] === 'ps' ? [] : { modelKey: 'k', identifier: 'i' }; } };
        const procDir = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), 'limits-lms-proc-'));
        t.after(() => fs.rmSync(procDir, { recursive: true, force: true }));
        // The start fails after the load, when it finds no engine process; the load is what is read.
        await runner.start(ctx, { procDir, kill: () => {}, node: '/usr/bin/node' }).catch(() => {});
        const override = loads[0].config.llamaCppArgumentsOverride.overrideParameters.find((entry) => entry.key === '--threads');
        return { threads: override.value, calls: calls.filter((call) => call.name === 'capThreads') };
    };
    const byDefault = await loadedThreads({});
    assert.equal(byDefault.threads, '3', 'the hook\'s answer is what the load asks for');
    assert.equal(byDefault.calls.length, 1);
    assert.equal(byDefault.calls[0].args[1].profile, 'dedicated');
    assert.deepEqual(byDefault.calls[0].args[1], { profile: 'dedicated' });
    const explicit = await loadedThreads({ threads: 9 });
    assert.equal(explicit.threads, '9');
    assert.equal(explicit.calls.length, 0, 'a caller-set value is not passed to capThreads');
});

const caps = (overrides = {}) => recording({
    capParams: ({ params, explicit }) => {
        if (params.ctxSize > 1024) {
            if (explicit.includes('ctxSize')) throw new LocalLlmError('invalid_params', `Invalid parameter ctxSize: ${params.ctxSize} is above the cap of 1024`, { field: 'ctxSize' });
            return { ...params, ctxSize: 1024 };
        }
        return params;
    },
    ...overrides,
});

// `machine.memory` is what the snapshot shows (a test may change it between admissions); `onDownload` runs when the weights are fetched.
async function paramsController(t, limits, { dataDir = null, startArgs = [], registry = null, weights = null, machine = null, onDownload = null, onSnapshot = null } = {}) {
    dataDir ??= fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), 'limits-params-'));
    if (!fs.existsSync(path.join(dataDir, 'weight.gguf'))) fs.writeFileSync(path.join(dataDir, 'weight.gguf'), weights ?? 'weights');
    if (registry) {
        fs.mkdirSync(path.join(dataDir, 'state'), { recursive: true });
        fs.writeFileSync(path.join(dataDir, 'state', 'controller.json'), JSON.stringify({ version: 1, deployment: null, params: {}, requests: {}, registry }));
    }
    const memory = { totalBytes: 8 * GIB, availableBytes: 7 * GIB };
    const seed = loadSeedCatalog(SEED_CATALOG_FILE).filter((entry) => entry.id === 'qwen2.5-0.5b-instruct-q4_k_m').map((entry) => validateModel(entry, { seed: true }));
    machine ??= { memory };
    const snapshot = { gpu: { available: false, state: 'absent' }, disk: { freeBytes: 1e12 } };
    const controller = createController({ dataDir, seedCatalog: seed, sharedModelsRoot: null, imageContract: {}, limits, env: { PATH: '/usr/bin' },
        fileExists: () => true, installer: { installable: () => false }, detectRunner: () => ({ installed: true }),
        snapshot: async () => { onSnapshot?.(); return { ...snapshot, memory: machine.memory }; }, readMemory: () => memory, readPressure: () => 0,
        inspect: async () => ({ state: 'complete', bytes: 7, total: 7 }),
        download: async () => { onDownload?.(); return { status: 'complete', path: path.join(dataDir, 'weight.gguf'), bytesTransferred: 0 }; },
        dropCache: () => true, pollMs: 2, stopGraceMs: 20, fetchImpl: async () => ({ ok: true, status: 200, json: async () => ({}) }),
        startRunner(options) {
            startArgs.push(options.args);
            let resolve;
            const handle = { pid: 9000, running: true, exited: new Promise((done) => { resolve = done; }),
                async stop() { handle.running = false; resolve({ code: 0, signal: 'SIGTERM' }); }, async kill() { return handle.stop(); } };
            return handle;
        } });
    t.after(async () => { await controller.drain(); });
    t.after(() => fs.rmSync(dataDir, { recursive: true, force: true }));
    return { controller, seed, dataDir };
}
const ctxOf = (args) => args[args.indexOf('--ctx-size') + 1];
const SAVED = 'qwen2.5-0.5b-instruct-q4_k_m|llama.cpp';

test('capParams: a default context is clamped, an explicit one above the cap is refused before launch, and a saved value is a default', async (t) => {
    // The control: no hook, the cpu default of 4096.
    const plainArgs = [];
    const plain = await paramsController(t, undefined, { startArgs: plainArgs });
    const id = plain.seed[0].id;
    await plain.controller.run({ modelId: id, runnerId: 'llama.cpp', requestId: 'cap-params-0001', params: {} });
    await until(() => plainArgs.length === 1);
    assert.equal(ctxOf(plainArgs[0]), '4096');

    const probe = caps();
    const args = [];
    const first = await paramsController(t, probe.limits, { startArgs: args });
    await first.controller.run({ modelId: id, runnerId: 'llama.cpp', requestId: 'cap-params-0002', params: {} });
    await until(() => args.length === 1);
    assert.equal(ctxOf(args[0]), '1024', 'the default context is clamped, in the launch');
    assert.equal(first.controller.state.deployment.params.ctxSize, 1024, 'and in the deployment');
    const [context] = probe.named('capParams')[0].args;
    assert.deepEqual({ ...context, params: undefined, model: undefined }, { runnerId: 'llama.cpp', profile: 'cpu', params: undefined, explicit: [], model: undefined });
    assert.equal(context.params.ctxSize, 4096, 'the hook got the normalized parameters, before the cap');
    assert.equal(context.model.id, id);
    assert.ok(Object.isFrozen(context.explicit));

    // The explicit value is told apart from the saved and the default one.
    await first.controller.stop();
    const asked = await first.controller.run({ modelId: id, runnerId: 'llama.cpp', requestId: 'cap-params-0003', params: { ctxSize: 2048 } }).catch((error) => error);
    assert.equal(asked.code, 'invalid_params', 'refused by Run itself, before anything is downloaded or started');
    assert.match(asked.message, /ctxSize: 2048 is above the cap of 1024/);
    assert.deepEqual(probe.named('capParams').at(-1).args[0].explicit, ['ctxSize']);
    assert.equal(args.length, 1, 'nothing was started');
    assert.equal(first.controller.state.deployment?.phase === 'error' || first.controller.state.deployment?.phase === 'ready', false);
    await assert.rejects(first.controller.overview({ preview: { modelId: id, runnerId: 'llama.cpp', params: { ctxSize: 2048 } } }), { code: 'invalid_params' }, 'and a preview throws the refusal, not a form error');
    // A parameter error of the runner itself stays form data in a preview.
    const form = await first.controller.overview({ preview: { modelId: id, runnerId: 'llama.cpp', params: { ctxSize: 1 } } });
    assert.match(form.preview.error, /ctxSize/);

    // A saved value above the cap is a default: Run with no params clamps it, and saved explicit-looking 4096 does not refuse.
    assert.equal(first.controller.state.params[SAVED].ctxSize, 4096);
    const savedProbe = caps();
    const second = await paramsController(t, savedProbe.limits, { dataDir: first.dataDir, startArgs: [] });
    assert.equal(second.controller.state.params[SAVED].ctxSize, 4096, 'the saved set was loaded');
    await second.controller.run({ modelId: id, runnerId: 'llama.cpp', requestId: 'cap-params-0004' });
    assert.equal(second.controller.state.deployment.params.ctxSize, 1024, 'the saved 4096 was clamped, not refused');
    assert.deepEqual(savedProbe.named('capParams').at(-1).args[0].explicit, []);
});

test('capParams: removing the cap restores the earlier parameters, because Run saves the set before the cap (G5)', async (t) => {
    const first = await paramsController(t, caps().limits, { startArgs: [] });
    const id = first.seed[0].id;
    await first.controller.run({ modelId: id, runnerId: 'llama.cpp', requestId: 'cap-saved-0001', params: {} });
    assert.equal(first.controller.state.deployment.params.ctxSize, 1024);
    assert.equal(first.controller.state.params[SAVED].ctxSize, 4096, 'the Run site saved the pre-cap set');
    await first.controller.drain();
    // The same state with the identity limits: the earlier value is back, in the preview and the Run.
    const args = [];
    const later = await paramsController(t, undefined, { dataDir: first.dataDir, startArgs: args });
    const preview = await later.controller.overview({ preview: { modelId: id, runnerId: 'llama.cpp' } });
    assert.equal(preview.preview.params.ctxSize, 4096);
    await later.controller.run({ modelId: id, runnerId: 'llama.cpp', requestId: 'cap-saved-0002' });
    await until(() => args.length === 1);
    assert.equal(ctxOf(args[0]), '4096');
});

test('capParams: the renormalize site saves the pre-cap set, and the deployment and the launch use the capped one', async (t) => {
    // A model added from a GGUF header: the stored training context is 32768 and the downloaded file says 2048, so the Run works out its parameters again.
    const header = (context) => {
        const bytes = ggufBytes({ kv: modelPairs({ layers: 24, heads: 14, kvHeads: 2, embedding: 896, context }) }).bytes;
        return Buffer.concat([bytes, Buffer.alloc(Math.max(0, 65536 - bytes.length), 0xee)]);
    };
    const entry = validateModel({
        id: 'trained-short', sizingSource: 'gguf-header', contextLength: 32768, memory: { layers: 24, kvBytesPerToken: 12288 },
        sources: { gguf: { type: 'huggingface', repo: 'acme/models', file: 'small-Q4_K_M.gguf', commit: 'c'.repeat(40), size: 400 * MIB, sha256: 'a'.repeat(64) } },
    });
    const probe = caps();
    const args = [];
    const { controller } = await paramsController(t, probe.limits, { startArgs: args, registry: [entry], weights: header(2048) });
    await controller.run({ modelId: 'trained-short', runnerId: 'llama.cpp', requestId: 'cap-renorm-0001' });
    await until(() => ['ready', 'error'].includes(controller.state.deployment?.phase));
    const { deployment } = controller.state;
    assert.equal(deployment.phase, 'ready', deployment.error);
    const called = probe.named('capParams');
    assert.equal(called.length >= 2, true, 'the Run, and the renormalize after the header changed the training context');
    assert.equal(called.at(-1).args[0].model.contextLength, 2048, 'the last call is the renormalize, with the verified model');
    assert.equal(called.at(-1).args[0].params.ctxSize, 2048, 'the hook got the pre-cap set: the default lowered to the training context');
    assert.deepEqual(called.at(-1).args[0].explicit, []);
    assert.equal(deployment.params.ctxSize, 1024, 'the deployment has the capped set');
    assert.equal(ctxOf(args[0]), '1024', 'and so has the launch');
    assert.equal(controller.state.params['trained-short|llama.cpp'].ctxSize, 2048, 'the renormalize site saved the pre-cap set');
});

test('capParams: a result outside the runner\'s schema is refused as invalid parameters, and a refusal the hook throws reaches the caller as it is', async (t) => {
    const bad = recording({ capParams: ({ params }) => ({ ...params, ctxSize: 3 }) });
    const { controller, seed } = await paramsController(t, bad.limits, { startArgs: [] });
    const id = seed[0].id;
    const refused = await controller.run({ modelId: id, runnerId: 'llama.cpp', requestId: 'cap-valid-0001' }).catch((error) => error);
    assert.equal(refused.code, 'invalid_params');
    assert.match(refused.message, /ctxSize: must be >= 512/);
    const unknown = recording({ capParams: ({ params }) => ({ ...params, bogus: 1 }) });
    const other = await paramsController(t, unknown.limits, { startArgs: [] });
    assert.match((await other.controller.run({ modelId: id, runnerId: 'llama.cpp', requestId: 'cap-valid-0002' }).catch((error) => error)).message, /bogus: unknown parameter/);
    assert.equal(other.controller.state.params[SAVED], undefined, 'nothing was saved by a refused Run');
    // A hook that returns an equal set keeps the engine's own behavior: the same values, validated.
    const same = recording({ capParams: ({ params }) => ({ ...params }) });
    const kept = await paramsController(t, same.limits, { startArgs: [] });
    await kept.controller.run({ modelId: id, runnerId: 'llama.cpp', requestId: 'cap-valid-0003' });
    assert.equal(kept.controller.state.deployment.params.ctxSize, 4096);
});

// ---------------------------------------------------------------- fitParams: fit.mjs on its own

const FIT_SCHEMA = LLAMA_SERVER_PARAM_SCHEMA;
const SIZED = (status, reason = 'too big', estimate = {}, code = null) => admissionResult(status, reason, { weightsBytes: MIB, ...estimate }, [], code);
// A refusal about the sizes names the need, which moves with them; this one does, as the policies' reasons do.
const needs = (set) => `Needs ${set.ctxSize} tokens and ${set.parallel} slots`;
// The GPU need the estimate carries: it grows with the context and the slots, as the KV cache does.
const gpuNeed = (set) => set.ctxSize * set.parallel * 1024;

/** fitAdmission with an `admitWith` that records every parameter set it is asked about and answers `verdict(params)`. */
// `code` is the reasonCode of each answer (the figure a llama-server refusal names), none by default.
function fitRun(params, { verdict = () => 'incompatible', explicit = [], order = FIT_KEYS, model = { id: 'm', memory: { layers: 24 } }, profile = 'dedicated', schema = FIT_SCHEMA, normalize = (value) => value, maxSteps, reason = needs, estimate = (set) => ({ gpuBytes: gpuNeed(set) }), code = () => null } = {}) {
    const asked = [];
    const result = fitAdmission({ params, explicit, order, schema, model, profile, normalize, ...(maxSteps ? { maxSteps } : {}),
        admitWith: (candidate) => { asked.push(candidate); return SIZED(verdict(candidate), reason(candidate), estimate(candidate), code(candidate)); } });
    return { ...result, asked };
}
const tuple = (set) => [set.ctxSize, set.parallel];
const START = Object.freeze({ ctxSize: 2048, parallel: 4, nGpuLayers: 99 });

test('fitParams: the identity does not fit, and a hook with no order is no fit', () => {
    assert.equal(IDENTITY_LIMITS.fitParams({ runnerId: 'llama.cpp', profile: 'dedicated' }), null);
    assert.deepEqual([...FIT_KEYS], ['ctxSize', 'nGpuLayers', 'parallel']);
    assert.equal(FIT_CONTEXT_FLOOR, 8192);
});

test('fit: the steps are ctxSize, then parallel, each halved to its end before the next is touched, and a refusal that names no figure leaves the layers alone', () => {
    const { asked, admission, params } = fitRun(START);
    // ctxSize 2048 -> 1024 -> 512 (the schema minimum); parallel 4 -> 2 -> 1.
    assert.deepEqual(asked.map(tuple), [[2048, 4], [1024, 4], [512, 4], [512, 2], [512, 1]]);
    assert.ok(asked.every((set) => set.nGpuLayers === 99), 'nGpuLayers is lowered only for a refusal that names the GPU');
    assert.equal(admission.status, 'incompatible');
    assert.equal(params, START, 'a request that does not fit is refused as it was asked');
    assert.match(admission.reason, /the smallest size tried was ctxSize 512, nGpuLayers 99, parallel 1\.$/, 'a refusal names the smallest size tried');
    assert.match(admission.reason, /^Needs 2048 tokens and 4 slots Fit also tried lowering ctxSize from 2048 to 512, parallel from 4 to 1, and it still does not fit;/);
    assert.equal(admission.fit, undefined, 'nothing is reported as lowered: nothing was');
    assert.equal(admission.warnings.length, 0);
});

test('fit: it stops at the first set that is not incompatible, which is the largest halved size that fits', () => {
    const { asked, admission, params } = fitRun(START, { verdict: (set) => (set.ctxSize <= 1024 ? 'ok' : 'incompatible') });
    assert.deepEqual(asked.map(tuple), [[2048, 4], [1024, 4]]);
    assert.equal(admission.status, 'ok');
    assert.deepEqual(tuple(params), [1024, 4]);
    assert.deepEqual(admission.fit, { lowered: [{ param: 'ctxSize', from: 2048, to: 1024 }] });
    assert.match(admission.warnings.at(-1), /^Fit lowered ctxSize from 2048 to 1024 so that the model fits/);
    assert.ok(Object.isFrozen(admission) && Object.isFrozen(admission.warnings) && Object.isFrozen(admission.fit));
    // A result that fits as it is comes back untouched, with no fit and no warning.
    const fits = fitRun(START, { verdict: () => 'ok' });
    assert.equal(fits.asked.length, 1);
    assert.equal(fits.params, START);
    assert.equal(fits.admission.fit, undefined);
    assert.deepEqual([...fits.admission.warnings], []);
});

test('fit: a key the caller set is never touched, and with every key set there is nothing to lower', () => {
    const ctx = fitRun(START, { explicit: ['ctxSize'] });
    assert.ok(ctx.asked.every((set) => set.ctxSize === 2048), 'ctxSize stays what the caller set in every candidate');
    assert.deepEqual(ctx.asked.map(tuple).slice(0, 3), [[2048, 4], [2048, 2], [2048, 1]], 'the next key in the order is lowered instead');
    for (const key of FIT_KEYS) {
        const run = fitRun(START, { explicit: [key] });
        assert.ok(run.asked.every((set) => set[key] === START[key]), key);
    }
    const all = fitRun(START, { explicit: [...FIT_KEYS] });
    assert.equal(all.asked.length, 1, 'only the request as it is was admitted');
    assert.equal(all.params, START);
    assert.equal(all.admission.status, 'incompatible');
    assert.equal(all.admission.reason, 'Needs 2048 tokens and 4 slots', 'with nothing to lower the refusal is left as it was');
    assert.equal(all.admission.fit, undefined);
});

test('fit: the number of steps is bounded, by the argument and by MAX_FIT_STEPS for a schema that never lets it end', () => {
    const three = fitRun(START, { maxSteps: 3 });
    assert.equal(three.asked.length, 1 + 3);
    assert.deepEqual(tuple(three.asked.at(-1)), [512, 2], 'ctxSize 1024, 512, then one step of parallel');
    // More steps to the minimum than the bound allows (52 halvings of the context, then one step per layer of 1000): the bound ends the fit.
    assert.equal(MAX_FIT_STEPS, 256);
    const schema = { properties: { ctxSize: { minimum: 1 }, nGpuLayers: { minimum: 0 } } };
    const endless = fitRun({ ctxSize: 2 ** 52, nGpuLayers: 1000 }, { order: ['nGpuLayers', 'ctxSize'], schema, model: { memory: { layers: 1000 } },
        code: () => 'gpu_capacity', estimate: (set) => ({ gpuBytes: set.nGpuLayers }), reason: (set) => `Needs ${set.nGpuLayers} layers` });
    assert.equal(endless.asked.length, 1 + MAX_FIT_STEPS);
    assert.equal(endless.asked.at(-1).ctxSize, 2 ** 52, 'the layers come first in this order');
    assert.equal(endless.asked.at(-1).nGpuLayers, 1000 - MAX_FIT_STEPS, 'one layer per step for every step there was');
    assert.match(endless.admission.reason, /the smallest size tried was nGpuLayers 744, ctxSize \d+\.$/);
});

test('fit: insufficient-now is never fitted, and neither is a refusal that no size caused', () => {
    const now = fitRun(START, { verdict: () => 'insufficient-now' });
    assert.equal(now.asked.length, 1);
    assert.equal(now.params, START);
    assert.equal(now.admission.status, 'insufficient-now');
    assert.equal(now.admission.fit, undefined);
    // An incompatible result whose estimate has no weights (the runner, the source or the machine said no) is not about the sizes.
    const unsized = fitAdmission({ params: START, explicit: [], order: FIT_KEYS, schema: FIT_SCHEMA, model: {}, profile: 'dedicated', normalize: (value) => value,
        admitWith: () => admissionResult('incompatible', 'No GPU is available.', {}) });
    assert.equal(unsized.params, START);
    assert.equal(unsized.admission.reason, 'No GPU is available.');
});

test('fit: a refusal that no size caused is returned as it was, with the parameters as asked, even when its estimate is full', () => {
    // The wording never moves with the sizes (a Metal device whose limit is unknown, memory that cannot be read): fit says nothing about it.
    const run = fitRun(START, { reason: () => 'The device limit is unknown, so nothing can be sized on it.' });
    assert.ok(run.asked.length > 1, 'it did try');
    assert.equal(run.params, START);
    assert.equal(run.admission.reason, 'The device limit is unknown, so nothing can be sized on it.');
    assert.equal(run.admission.fit, undefined);
    assert.deepEqual([...run.admission.warnings], []);
});

test('fit: the order is the hook\'s, and a key it does not name is not lowered', () => {
    const reversed = fitRun(START, { order: ['parallel', 'ctxSize'] });
    assert.deepEqual(reversed.asked.map(tuple).slice(0, 4), [[2048, 4], [2048, 2], [2048, 1], [1024, 1]]);
    assert.ok(reversed.asked.every((set) => set.nGpuLayers === 99), 'the layers stay as asked');
    const unknown = fitRun(START, { order: ['bogus', 'ctxSize'] });
    assert.ok(unknown.asked.every((set) => set.parallel === 4 && set.nGpuLayers === 99));
    const layers = fitRun(START, { order: ['nGpuLayers'] });
    assert.equal(layers.asked.length, 1, 'the layers move only for a refusal that names the GPU, and this one names none');
    const gpu = fitRun(START, { order: ['parallel'], code: () => 'gpu_capacity' });
    assert.ok(gpu.asked.every((set) => set.nGpuLayers === 99 && set.ctxSize === 2048), 'a GPU refusal lowers only what the order names');
});

test('fit: nGpuLayers is not lowered for a refusal that names no figure, on any profile, with or without a layer count', () => {
    for (const [profile, model] of [['dedicated', { id: 'm' }], ['dedicated', { id: 'm', memory: { layers: 24 } }], ['cpu', { id: 'm' }]]) {
        const run = fitRun({ ctxSize: 512, parallel: 1, nGpuLayers: 99 }, { model, profile });
        assert.equal(run.asked.length, 1, `${profile}: nothing else could be lowered, and the layers are not`);
    }
    const mixed = fitRun({ ctxSize: 2048, parallel: 1, nGpuLayers: 20 });
    assert.ok(mixed.asked.every((set) => set.nGpuLayers === 20), 'a partial offload keeps its layers');
});

test('fit: the wording of a result that fits only this machine\'s size says it does not fit what is free now', () => {
    const run = fitRun(START, { verdict: (set) => (set.ctxSize <= 1024 ? 'insufficient-now' : 'incompatible') });
    assert.equal(run.admission.status, 'insufficient-now');
    assert.deepEqual(tuple(run.params), [1024, 4]);
    assert.equal(run.admission.warnings.at(-1), 'Fit lowered ctxSize from 2048 to 1024 so that the model fits this machine\'s size; it does not fit in what is free now.');
    assert.ok(!/so that the model fits;/.test(run.admission.warnings.at(-1)));
    assert.deepEqual(run.admission.fit, { lowered: [{ param: 'ctxSize', from: 2048, to: 1024 }] });
});

test('fit: a refusal whose need did not go down, though its wording moved, gets no "Fit also tried" sentence', () => {
    // A RAM refusal: lowering the context takes the GPU figure down and leaves the RAM figure where it was.
    const flat = fitRun(START, { reason: () => 'Needs about 40 GiB of RAM; this machine has 32 GiB.', estimate: (set) => ({ gpuBytes: gpuNeed(set), ramBytes: 40 * MIB }) });
    assert.ok(flat.asked.length > 1, 'it did try');
    assert.equal(flat.admission.reason, 'Needs about 40 GiB of RAM; this machine has 32 GiB.');
    // The wording moves but no figure went down (the estimate carries none that does): the need did not go down, so nothing is added.
    const moved = fitRun(START, { estimate: () => ({ gpuBytes: 7 * MIB, ramBytes: 7 * MIB }) });
    assert.equal(moved.admission.reason, 'Needs 2048 tokens and 4 slots');
    // Control: the GPU need goes down with the sizes and the wording follows it, so the sentence is added.
    const control = fitRun(START);
    assert.match(control.admission.reason, /Fit also tried lowering/);
});

// A llama-server-like answer: the GPU need is the layers' share of the weights and of the KV cache; the RAM need is the rest.
const LAYERED = { id: 'm', memory: { layers: 40 } };
const layerShare = (set) => Math.min(set.nGpuLayers, 40) / 40;
const kvOf = (set) => set.ctxSize * set.parallel * 1024;
const layeredNeed = (set) => ({ gpuBytes: Math.round((14 * GIB + kvOf(set)) * layerShare(set)), ramBytes: Math.round((14 * GIB + kvOf(set)) * (1 - layerShare(set))) });
const layeredReason = (set) => `GPU ${layeredNeed(set).gpuBytes} RAM ${layeredNeed(set).ramBytes}`;
const tri = (set) => [set.ctxSize, set.nGpuLayers, set.parallel];

test('fit: a GPU refusal takes a default context down to the floor, then moves the layers one at a time, and stops at the first that fits', () => {
    const run = fitRun({ ctxSize: 32768, parallel: 1, nGpuLayers: 99 }, { model: LAYERED, reason: layeredReason, estimate: layeredNeed,
        code: () => 'gpu_capacity', verdict: (set) => (layeredNeed(set).gpuBytes <= 7 * GIB ? 'ok' : 'incompatible') });
    assert.deepEqual(run.asked.slice(0, 4).map(tri), [[32768, 99, 1], [16384, 99, 1], [8192, 99, 1], [8192, 39, 1]],
        'the context halves to the floor, then the first step goes from 99 (all) to one below the model\'s 40 layers');
    assert.ok(run.asked.every((set) => set.ctxSize >= FIT_CONTEXT_FLOOR), 'the context never went below the floor: the layers were enough');
    assert.equal(run.admission.status, 'ok');
    assert.deepEqual(tri(run.params), [8192, 19, 1], 'the largest layer count that fits: 19 of 40 layers of 14 GiB and the KV cache is under 7 GiB');
    assert.deepEqual(run.admission.fit, { lowered: [{ param: 'ctxSize', from: 32768, to: 8192 }, { param: 'nGpuLayers', from: 99, to: 19 }] });
    assert.match(run.admission.warnings.at(-1), /^Fit lowered ctxSize from 32768 to 8192, nGpuLayers from 99 to 19 so that the model fits/);
    // A context at or under the floor is not raised and does not move before the layers.
    const small = fitRun({ ctxSize: 4096, parallel: 1, nGpuLayers: 99 }, { model: LAYERED, reason: layeredReason, estimate: layeredNeed,
        code: () => 'gpu_capacity', verdict: (set) => (layeredNeed(set).gpuBytes <= 7 * GIB ? 'ok' : 'incompatible') });
    assert.ok(small.asked.every((set) => set.ctxSize === 4096));
    assert.equal(small.params.nGpuLayers, 19);
    // A context the caller set stays; the layers move all the same.
    const explicit = fitRun({ ctxSize: 32768, parallel: 1, nGpuLayers: 99 }, { model: LAYERED, reason: layeredReason, estimate: layeredNeed, explicit: ['ctxSize'],
        code: () => 'gpu_capacity', verdict: (set) => (layeredNeed(set).gpuBytes <= 7 * GIB ? 'ok' : 'incompatible') });
    assert.deepEqual(tri(explicit.params), [32768, 19, 1]);
    // Layers the caller set stay, and the context is not held at the floor for them: it goes on down.
    const pinned = fitRun({ ctxSize: 32768, parallel: 1, nGpuLayers: 99 }, { model: LAYERED, explicit: ['nGpuLayers'], code: () => 'gpu_capacity' });
    assert.ok(pinned.asked.every((set) => set.nGpuLayers === 99));
    assert.equal(pinned.asked.at(-1).ctxSize, 512, 'with no layers to move, the context goes to the schema minimum');
});

test('fit: the layers stop at 1, never 0, and then the context goes on below the floor; a model with no layer count is sized with 48', () => {
    const run = fitRun({ ctxSize: 16384, parallel: 2, nGpuLayers: 99 }, { model: LAYERED, code: () => 'gpu_capacity' });
    const layersTried = run.asked.map((set) => set.nGpuLayers);
    assert.equal(Math.min(...layersTried), 1, 'one layer stays on the GPU: at 0 the estimate counts no compute buffer and no CUDA context');
    assert.deepEqual(run.asked.slice(1, 3).map(tri), [[8192, 99, 2], [8192, 39, 2]]);
    assert.deepEqual(tri(run.asked.at(-1)), [512, 1, 1], 'then the context to its minimum, then the slots');
    assert.equal(run.asked.length, 1 + 1 + 39 + 4 + 1, 'the floor, 39 layer steps (40 to 1), four halvings of the context and one of the slots');
    const unknown = fitRun({ ctxSize: 8192, parallel: 1, nGpuLayers: 99 }, { model: { id: 'm' }, code: () => 'gpu_capacity' });
    assert.equal(unknown.asked[1].nGpuLayers, 47, 'one below the 48 layers the estimate assumes');
    // A model already at one layer has none to move.
    const one = fitRun({ ctxSize: 8192, parallel: 1, nGpuLayers: 1 }, { model: LAYERED, code: () => 'gpu_capacity' });
    assert.ok(one.asked.every((set) => set.nGpuLayers === 1));
});

test('fit: the layers move only while the refusal names the GPU; once the RAM is the refusal the context goes on below the floor', () => {
    // 14 GiB of weights and 512 KiB of KV cache per token; the GPU holds 6 GiB, the RAM 10 GiB. At the 8192 floor 13 layers fit the GPU,
    // and the other 27 need 12.2 GiB of RAM: the GPU refusal has turned into a RAM one, and the context then goes down until both fit.
    const heavy = (set) => {
        const total = 14 * GIB + set.ctxSize * set.parallel * 512 * 1024;
        return { gpuBytes: Math.round(total * layerShare(set)), ramBytes: Math.round(total * (1 - layerShare(set))) };
    };
    const fitsGpu = (set) => heavy(set).gpuBytes <= 6 * GIB;
    const fitsRam = (set) => heavy(set).ramBytes <= 10 * GIB;
    const run = fitRun({ ctxSize: 16384, parallel: 1, nGpuLayers: 99 }, { model: LAYERED, reason: (set) => JSON.stringify(heavy(set)), estimate: heavy,
        code: (set) => (!fitsGpu(set) ? 'gpu_capacity' : !fitsRam(set) ? 'ram_capacity' : null),
        verdict: (set) => (fitsGpu(set) && fitsRam(set) ? 'ok' : 'incompatible') });
    const firstRam = run.asked.findIndex((set) => fitsGpu(set) && !fitsRam(set));
    assert.ok(firstRam > 0, 'the walk reached a RAM refusal');
    assert.ok(run.asked.slice(firstRam + 1).every((set) => set.nGpuLayers === run.asked[firstRam].nGpuLayers), 'no layer moved after it');
    assert.equal(run.asked[firstRam].nGpuLayers, 13);
    assert.equal(run.admission.status, 'ok', run.admission.reason);
    assert.deepEqual(tri(run.params), [1024, 13, 1], 'the context went below the floor, with the layers kept');
    // A RAM refusal from the start never moves a layer.
    const ram = fitRun({ ctxSize: 16384, parallel: 1, nGpuLayers: 99 }, { model: LAYERED, code: () => 'ram_capacity' });
    assert.ok(ram.asked.every((set) => set.nGpuLayers === 99));
    assert.equal(ram.asked.at(-1).ctxSize, 512, 'and the context is not held at the floor');
});

test('fit: the llama-server policy names the figure of its incompatible refusals in the reasonCode fit reads, and LM Studio\'s too', () => {
    const model = { id: 'm', memory: { layers: 40 } };
    const params = { ctxSize: 16384, nGpuLayers: 99, nCpuMoe: 0, flashAttn: 'auto', cacheTypeK: 'f16', cacheTypeV: 'f16', parallel: 1, ubatchSize: 512 };
    const machine = (gpuGiB, ramGiB) => ({ gpu: { totalBytes: gpuGiB * GIB, freeBytes: gpuGiB * GIB }, memory: { totalBytes: ramGiB * GIB, availableBytes: ramGiB * GIB }, disk: {} });
    for (const policy of [admitLlamaServer, admitLmStudio]) {
        const gpu = policy({ model, source: { size: 14 * GIB }, params, ...machine(8, 64) });
        assert.deepEqual([gpu.status, gpu.reasonCode], ['incompatible', 'gpu_capacity'], gpu.reason);
        const ram = policy({ model, source: { size: 14 * GIB }, params: { ...params, nGpuLayers: 10 }, ...machine(24, 8) });
        assert.deepEqual([ram.status, ram.reasonCode], ['incompatible', 'ram_capacity'], ram.reason);
        const ok = policy({ model, source: { size: 2 * GIB }, params, ...machine(24, 64) });
        assert.deepEqual([ok.status, ok.reasonCode], ['ok', undefined]);
    }
});

test('fit: "went down" is judged by the figure the first refusal names', () => {
    // A RAM refusal whose RAM figure the context does not move, while the GPU figure goes down with it: no "Fit also tried" sentence.
    const ram = fitRun(START, { reason: needs, code: () => 'ram_capacity', estimate: (set) => ({ gpuBytes: gpuNeed(set), ramBytes: 40 * GIB }) });
    assert.ok(ram.asked.length > 1, 'it did try');
    assert.equal(ram.admission.reason, 'Needs 2048 tokens and 4 slots');
    // A GPU refusal walked only through the layers: the GPU figure goes down and the RAM figure up; the refusal says what fit tried.
    const pinned = { ctxSize: 16384, parallel: 1, nGpuLayers: 99 };
    const gpu = fitRun(pinned, { model: LAYERED, reason: layeredReason, estimate: layeredNeed, explicit: ['ctxSize', 'parallel'], code: () => 'gpu_capacity' });
    assert.ok(gpu.asked.every((set) => set.ctxSize === 16384 && set.parallel === 1));
    assert.ok(layeredNeed(gpu.asked.at(-1)).ramBytes > layeredNeed(pinned).ramBytes, 'the RAM figure rose');
    assert.equal(gpu.params, pinned);
    assert.match(gpu.admission.reason, /^GPU \d+ RAM \d+ Fit also tried lowering nGpuLayers from 99 to 1, and it still does not fit; the smallest size tried was ctxSize 16384, nGpuLayers 1, parallel 1\.$/);
    // The same walk ending in a RAM refusal is judged by the GPU figure the first refusal named, which went down.
    const turned = fitRun(pinned, { model: LAYERED, reason: layeredReason, estimate: layeredNeed, explicit: ['ctxSize', 'parallel'],
        code: (set) => (set.nGpuLayers > 20 ? 'gpu_capacity' : 'ram_capacity') });
    assert.equal(turned.asked.at(-1).nGpuLayers, 20, 'the layers stopped where the RAM became the refusal');
    assert.match(turned.admission.reason, /Fit also tried lowering nGpuLayers from 99 to 20, and it still does not fit;/);
});

test('fit: lowering parallel lowers a context the training context per slot cannot hold, unless the caller set it', () => {
    const model = { id: 'm', contextLength: 1024, memory: { layers: 24 } };
    const lowered = fitRun({ ctxSize: 4096, parallel: 4, nGpuLayers: 1 }, { order: ['parallel'], model });
    assert.deepEqual(lowered.asked.map(tuple), [[4096, 4], [2048, 2], [1024, 1]]);
    const set = fitRun({ ctxSize: 4096, parallel: 4, nGpuLayers: 1 }, { order: ['parallel'], model, explicit: ['ctxSize'] });
    assert.equal(set.asked.length, 1, 'the slots that could not hold the caller\'s context are not lowered');
    // A candidate the schema refuses cannot be lowered either, and the next key is tried; another error is a fault.
    const refusing = fitRun(START, { normalize: (candidate) => { if (candidate.ctxSize < 2048) throw Object.assign(new Error('no'), { code: 'invalid_params' }); return candidate; } });
    assert.ok(refusing.asked.slice(1).every((set) => set.ctxSize === 2048), 'ctxSize cannot go lower, so parallel is');
    assert.throws(() => fitRun(START, { normalize: () => { throw new Error('boom'); } }), /boom/);
});

// ---------------------------------------------------------------- fitParams: through the controller

// The cpu estimate of the seed model at a context, read from a preview on a roomy machine, and the memory of a machine whose pool holds exactly `ctx`.
async function cpuNeeds(t) {
    const { controller, seed } = await paramsController(t, undefined, { startArgs: [] });
    const id = seed[0].id;
    const need = async (ctxSize) => (await controller.overview({ preview: { modelId: id, runnerId: 'llama.cpp', params: { ctxSize } } })).preview.admission.estimate.ramBytes;
    return { id, need, machineFor: async (ctxSize) => ({ totalBytes: Math.ceil(1.5 * GIB) + (await need(ctxSize)) + MIB, availableBytes: Math.ceil(1.5 * GIB) + (await need(ctxSize)) + MIB }) };
}
const FIT_CTX = () => recording({ fitParams: () => ({ order: ['ctxSize', 'parallel'] }) });

test('fit through the controller: a preview and a Run use the largest halved context that fits, the saved set is the one before, and the status shows the chosen context', async (t) => {
    const { id, machineFor, need } = await cpuNeeds(t);
    assert.ok(await need(4096) > await need(2048) && await need(2048) > await need(1024), 'the estimate grows with the context');
    const machine = { memory: await machineFor(2048) };
    // The control: the identity on the same machine refuses the default context, and nothing is lowered or reported.
    const plain = await paramsController(t, undefined, { startArgs: [], machine });
    const refused = (await plain.controller.overview({ preview: { modelId: id, runnerId: 'llama.cpp' } })).preview;
    assert.equal(refused.admission.status, 'incompatible', refused.admission.reason);
    assert.equal(refused.params.ctxSize, 4096);
    assert.equal(refused.admission.fit, undefined);
    assert.ok(!/Fit/.test(refused.admission.reason));
    assert.deepEqual(await plain.controller.run({ modelId: id, runnerId: 'llama.cpp', requestId: 'fit-plain-0001' }).catch((error) => error.code), 'admission_incompatible');

    const probe = FIT_CTX();
    const args = [];
    const { controller } = await paramsController(t, probe.limits, { startArgs: args, machine });
    const { preview } = await controller.overview({ preview: { modelId: id, runnerId: 'llama.cpp' } });
    assert.equal(preview.admission.status, 'ok', preview.admission.reason);
    assert.equal(preview.params.ctxSize, 2048, 'the largest halved context that fits');
    assert.equal(preview.context.totalContext, 2048, 'the context shown is the fitted one');
    assert.deepEqual(preview.admission.fit, { lowered: [{ param: 'ctxSize', from: 4096, to: 2048 }] });
    assert.match(preview.admission.warnings.at(-1), /Fit lowered ctxSize from 4096 to 2048/);
    assert.equal(probe.named('fitParams').length, 2, 'the preview and the overview row ask once each');
    assert.deepEqual(probe.named('fitParams')[0].args[0], { runnerId: 'llama.cpp', profile: 'cpu' });
    // The overview's own row says the same: it is an admission point too.
    const overview = await controller.overview();
    const row = overview.models.find((model) => model.id === id).runners['llama.cpp'];
    assert.equal(row.admission.status, 'ok');
    assert.equal(row.params.ctxSize, 2048);

    await controller.run({ modelId: id, runnerId: 'llama.cpp', requestId: 'fit-run-0001' });
    await until(() => args.length === 1);
    assert.equal(ctxOf(args[0]), '2048', 'the launch uses the fitted context');
    const { deployment } = controller.state;
    assert.equal(deployment.params.ctxSize, 2048);
    assert.deepEqual(deployment.admission.fit, { lowered: [{ param: 'ctxSize', from: 4096, to: 2048 }] });
    assert.equal(controller.state.params[SAVED].ctxSize, 4096, 'what is saved is the set before fitting');
    const status = await controller.status();
    assert.equal(status.context.totalContext, 2048, 'the status shows the chosen context');
    assert.match(status.deployment.admission.warnings.at(-1), /Fit lowered/);
});

test('fit through the controller: a context the caller set is refused as it was, and insufficient-now is not fitted', async (t) => {
    const { id, machineFor } = await cpuNeeds(t);
    const machine = { memory: await machineFor(2048) };
    const { controller } = await paramsController(t, FIT_CTX().limits, { startArgs: [], machine });
    const asked = (await controller.overview({ preview: { modelId: id, runnerId: 'llama.cpp', params: { ctxSize: 4096 } } })).preview;
    assert.equal(asked.admission.status, 'incompatible');
    assert.equal(asked.params.ctxSize, 4096, 'the caller\'s value is untouched');
    assert.ok(!/Fit/.test(asked.admission.reason), 'the context is the caller\'s and the slots are at 1: nothing could be lowered, so the refusal is as it was');
    const run = await controller.run({ modelId: id, runnerId: 'llama.cpp', requestId: 'fit-explicit-0001', params: { ctxSize: 4096 } }).catch((error) => error);
    assert.equal(run.code, 'admission_incompatible');
    assert.equal(controller.state.params[SAVED], undefined, 'a refused Run saves nothing');
    // Memory that fits the pool but not what is available now: insufficient-now, which fit leaves as it is (a smaller context would fit).
    const need4096 = (await controller.overview({ preview: { modelId: id, runnerId: 'llama.cpp', params: { ctxSize: 4096 } } })).preview.admission.estimate.ramBytes;
    machine.memory = { totalBytes: 8 * GIB, availableBytes: cpuFloorBytes(8 * GIB) + need4096 - MIB };
    const probe = FIT_CTX();
    const busy = await paramsController(t, probe.limits, { startArgs: [], machine });
    const now = (await busy.controller.overview({ preview: { modelId: id, runnerId: 'llama.cpp', params: {} } })).preview;
    assert.equal(now.admission.status, 'insufficient-now', now.admission.reason);
    assert.equal(now.params.ctxSize, 4096, 'not lowered');
    assert.equal(now.admission.fit, undefined);
});

test('fit through the controller: the check after the download fits again, from the set the Run started with, on the memory then', async (t) => {
    const { id, machineFor } = await cpuNeeds(t);
    const machine = { memory: await machineFor(2048) };
    const tighter = await machineFor(1024);
    const probe = FIT_CTX();
    const args = [];
    // The weights are fetched after the Run was admitted at 2048; the machine has less memory by then.
    const { controller } = await paramsController(t, probe.limits, { startArgs: args, machine, onDownload: () => { machine.memory = tighter; } });
    await controller.run({ modelId: id, runnerId: 'llama.cpp', requestId: 'fit-recheck-0001' });
    await until(() => args.length === 1);
    assert.equal(controller.state.deployment.admission.fit.lowered[0].to, 1024, 'the recheck lowered it again');
    assert.equal(ctxOf(args[0]), '1024', 'and the launch uses the rechecked context');
    assert.equal(controller.state.deployment.params.ctxSize, 1024);
    assert.equal(controller.state.params[SAVED].ctxSize, 4096, 'the saved set is still the one before');
    // The Run itself and both of its repeated checks (after the download, and after the runner is prepared) asked for a fit.
    assert.equal(probe.named('fitParams').length, 3, 'the Run and its two repeated checks ask once each');
});

test('identity limits: every admission point makes exactly one admission from one snapshot, and asks the fit hook once', async (t) => {
    const probe = recording();
    let snaps = 0;
    const at = () => ({ admits: probe.named('admit').filter((call) => call.args[0].stage === 'before-policy').length, snaps, fits: probe.named('fitParams').length });
    const since = (from) => { const now = at(); return { admits: now.admits - from.admits, snaps: now.snaps - from.snaps, fits: now.fits - from.fits }; };
    let beforeDownload = null;
    const args = [];
    const { controller, seed } = await paramsController(t, probe.limits, { startArgs: args, onSnapshot: () => { snaps += 1; }, onDownload: () => { beforeDownload = at(); } });
    const id = seed[0].id;
    // The overview: one row is admitted (the other runners are not offered on this machine), from one snapshot.
    let from = at();
    await controller.overview();
    assert.deepEqual(since(from), { admits: 1, snaps: 1, fits: 1 }, 'the overview');
    // A preview adds its own admission to the overview's row, from the same one snapshot.
    from = at();
    await controller.overview({ preview: { modelId: id, runnerId: 'llama.cpp' } });
    assert.deepEqual(since(from), { admits: 2, snaps: 1, fits: 2 }, 'the preview and the row');
    // A Run: its own admission before the download, then the check after the download and the check after the runner is prepared.
    from = at();
    await controller.run({ modelId: id, runnerId: 'llama.cpp', requestId: 'one-admission-0001' });
    await until(() => args.length === 1);
    assert.deepEqual(since(beforeDownload), { admits: 2, snaps: 2, fits: 2 }, 'the check after the download and the check after the runner is prepared: one each');
    assert.deepEqual(since(from), { admits: 3, snaps: 3, fits: 3 }, 'the Run in all: one admission and one snapshot per point');
});
