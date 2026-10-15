// The limits hooks (slice C of the engine move): the identity default, and for each hook the proof that the engine calls it
// at its touchpoint with the documented inputs and uses what it returns. A recording fake plug-in stands in for the
// deployment that would bind real limits; every assertion about a result is about what the engine does with the fake's
// answer. The guard hooks are driven through the pooled guard in apu-profile.test.mjs.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { admissionResult, admit, admitOllama, admitVllm } from '../../lib/local-llm/controller/admission.mjs';
import { loadSeedCatalog, validateModel } from '../../lib/local-llm/controller/catalog.mjs';
import { cpuDefaultThreads, defaultThreads, performanceCoreCount, physicalCoreCount, readSnapshot } from '../../lib/local-llm/controller/hardware.mjs';
import { admitApuLlamaServer } from '../../lib/local-llm/controller/profiles.mjs';
import { LocalLlmError } from '../../lib/local-llm/errors.mjs';
import { IDENTITY_LIMITS, LIMIT_HOOKS, defineLimits } from '../../lib/local-llm/limits.mjs';
import { createRunners } from '../../lib/local-llm/runners/index.mjs';
import { createStateStore } from '../../lib/local-llm/controller/stateStore.mjs';
import { SEED_CATALOG_FILE } from './fixtures/seedCatalog.mjs';
import { HOST, PLOINKY_MESSAGES, RUNNERS, createController, getRunner } from './fixtures/host.mjs';
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
    assert.deepEqual(LIMIT_HOOKS, ['capCores', 'decorateSnapshot', 'memoryPool', 'refuseProfile', 'admit', 'effective',
        'refuseResourceRequest', 'overview', 'runnerEnvironment', 'guardSample', 'guardBreach', 'vllmQualification']);
    assert.ok(Object.isFrozen(IDENTITY_LIMITS));
    assert.equal(IDENTITY_LIMITS.capCores(7, { fsApi: {} }), 7);
    const snapshot = { gpu: {}, memory: { totalBytes: 1 } };
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
    assert.ok(calls.every((call) => call.name === 'capCores'));
    assert.ok(calls.length >= 2);
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

function onCpu(limits, { memory = { totalBytes: 8 * GIB, availableBytes: 7 * GIB } } = {}) {
    const snapshot = { gpu: { available: false, state: 'absent' }, memory, disk: { freeBytes: 100 * GIB } };
    const result = admit({ runner: OLLAMA, model: OLLAMA_MODEL, source: { type: 'ollama', tag: 't', size: 400 * MIB },
        params: OLLAMA.normalizeParams({}, { model: OLLAMA_MODEL, profile: 'cpu' }), snapshot, profile: 'cpu',
        decision: { cause: 'absent', reason: 'No GPU.' }, limits });
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
    // Memory that cannot be read is the engine's own refusal; the hook is not asked.
    const unread = recording({ memoryPool: pool });
    const refused = onCpu(unread.limits, { memory: {} }).result;
    assert.equal(refused.status, 'incompatible');
    assert.match(refused.reason, /cannot be read/);
    assert.equal(unread.calls.filter((call) => call.name === 'memoryPool').length, 0);
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

function dedicatedGuard(t, limits) {
    const dataDir = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), 'limits-guard-'));
    t.after(() => fs.rmSync(dataDir, { recursive: true, force: true }));
    const reads = { count: 0, availableBytes: 24 * GIB };
    const controller = createController({
        dataDir, env: { PATH: '/usr/bin' }, seedCatalog: loadSeedCatalog(SEED_CATALOG_FILE), stateStore: createStateStore({ dataDir }), limits,
        runners: { floored: FLOORED_RUNNER }, snapshot: async () => structuredClone(DEDICATED_SNAPSHOT),
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
    const run = (requestId) => controller.run({ modelId: 'gpt-oss-20b', runnerId: 'floored', requestId, params: {} });
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
