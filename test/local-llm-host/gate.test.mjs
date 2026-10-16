// The gate of the local-llm provider: what a request may start, join, replace or stop. Each test has its own temporary home, a
// real controller and the stand-in runner on a real Unix socket (fixtures/fake-runner.mjs); the clock is injected where an idle stop is timed.
import '../home.mjs';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';

import { createLocalLlmGate, defaultRunnerFor } from '../../lib/local-llm-host/gate.mjs';
import { SEED_CATALOG_FILE } from '../../lib/local-llm-host/host.mjs';
import { MODEL, runModel, scratch, until } from './fixtures/engine.mjs';
import { startHost } from './fixtures/proxy.mjs';

const ENTRY = { model: MODEL, runner: 'fake-gguf' };
const onDemand = (extra = {}) => ({ onDemand: { enabled: true, idleStopMs: 60_000, startWaitMs: 10_000, ...extra } });
const alive = (pid) => { try { process.kill(pid, 0); return true; } catch { return false; } };

test('without a model ready and without on-demand the gate refuses with the command to run one, and creates no engine', async (t) => {
    const { host } = await startHost(t);
    const gate = createLocalLlmGate({ host, settings: {} });
    const verdict = await gate.ensure(ENTRY);
    assert.equal(verdict.ok, false);
    assert.match(verdict.reason, /pworker models run/);
    assert.equal(host.current(), null, 'a refusal needs no controller');
    assert.equal(gate.status().last_refusal.reason, verdict.reason);
    assert.equal(gate.status().ready, false);
});

test('a refusal says why when the host is disabled, unsupported or has no model id to look for', async (t) => {
    const { host } = await startHost(t, { config: { localLlm: { enabled: false } } });
    assert.match((await createLocalLlmGate({ host }).ensure(ENTRY)).reason, /disabled/);
    const unsupported = (await import('../../lib/local-llm-host/host.mjs')).createLocalLlmHost({ home: scratch(t), platformName: 'darwin' });
    assert.match((await createLocalLlmGate({ host: unsupported }).ensure(ENTRY)).reason, /not supported on darwin/);
    assert.match((await createLocalLlmGate({ host: null }).ensure(ENTRY)).reason, /not available/);
    const running = await startHost(t);
    assert.match((await createLocalLlmGate({ host: running.host }).ensure({})).reason, /model id is required/);
});

test('an admin-run deployment is used as it is: the target names this start\'s socket and key', async (t) => {
    const { host } = await startHost(t);
    const controller = await host.controller();
    await runModel(controller);
    const gate = createLocalLlmGate({ host, settings: {} });
    const verdict = await gate.ensure(ENTRY);
    assert.equal(verdict.ok, true);
    assert.equal(verdict.target.modelId, MODEL);
    assert.equal(verdict.target.baseUrl, 'http://localhost');
    assert.ok(path.isAbsolute(verdict.target.socketPath) && fs.existsSync(verdict.target.socketPath));
    assert.ok(verdict.target.apiKey && verdict.target.apiKey.length >= 16);
    assert.equal(gate.status().startedBy, 'admin');
    assert.equal(gate.status().ready, true);
    // A request that names another runner is not served by this deployment.
    assert.equal((await gate.ensure({ model: MODEL, runner: 'llama.cpp' })).ok, false);
});

test('an admin-run deployment is never replaced and never idle-stopped, whatever the on-demand settings say', async (t) => {
    const { host } = await startHost(t);
    const controller = await host.controller();
    await runModel(controller);
    let clock = Date.now();
    const gate = createLocalLlmGate({ host, settings: onDemand({ idleStopMs: 50 }), now: () => clock });
    const other = await gate.ensure({ model: 'some-other-model' });
    assert.equal(other.ok, false);
    assert.match(other.reason, /admin-run deployment .* is active/);
    assert.equal((await gate.ensure(ENTRY)).ok, true);
    clock += 3_600_000;
    assert.deepEqual(await gate.idleCheck(), { stopped: false, reason: 'not the gate\'s deployment' });
    assert.equal(controller.state.deployment.phase, 'ready');
    assert.equal(gate.status().startedBy, 'admin');
});

test('on demand: 20 parallel requests start exactly one run and all get the same target', async (t) => {
    const { host, counter } = await startHost(t);
    const gate = createLocalLlmGate({ host, settings: onDemand(), pollMs: 10 });
    const verdicts = await Promise.all(Array.from({ length: 20 }, () => gate.ensure(ENTRY)));
    assert.ok(verdicts.every((verdict) => verdict.ok), JSON.stringify(verdicts.find((verdict) => !verdict.ok)));
    assert.equal(counter.starts, 1, 'one run for twenty requests');
    assert.equal(new Set(verdicts.map((verdict) => verdict.target.apiKey)).size, 1);
    assert.equal(gate.status().startedBy, 'on-demand');
    assert.equal(host.current().state.deployment.download?.transferred ?? 0, 0, 'no byte was fetched');
    // A later request finds the deployment ready and starts nothing.
    assert.equal((await gate.ensure(ENTRY)).ok, true);
    assert.equal(counter.starts, 1);
});

test('on demand never downloads: weights that are not on disk give a refusal and no run', async (t) => {
    const { host, counter } = await startHost(t, { weights: 'absent' });
    const gate = createLocalLlmGate({ host, settings: onDemand(), pollMs: 10 });
    const verdict = await gate.ensure(ENTRY);
    assert.equal(verdict.ok, false);
    assert.match(verdict.reason, /weights of .* are not downloaded/);
    assert.equal(counter.starts, 0);
    assert.equal(counter.downloads, 0);
    assert.equal(host.current().state.deployment, null, 'no deployment was recorded');
});

test('on demand: a run that finds its weights gone and starts fetching them is cancelled, so a request never downloads', async (t) => {
    // The weights look complete when the gate asks and are fetched when the run reads them: the second download progress report is bytes.
    const extra = { download: async ({ signal, onProgress }) => {
        onProgress({ bytes: 5, total: 100, rate: 1, transferred: 5, phase: 'downloading' });
        await new Promise((resolve) => signal.addEventListener('abort', resolve, { once: true }));
        throw Object.assign(new Error('aborted'), { code: 'ABORTED' });
    } };
    const { host, counter } = await startHost(t, { extra });
    const gate = createLocalLlmGate({ host, settings: onDemand(), pollMs: 10 });
    const verdict = await gate.ensure(ENTRY);
    assert.equal(verdict.ok, false);
    assert.match(verdict.reason, /are not downloaded/);
    assert.equal(counter.starts, 0, 'no runner was started');
    await until(() => host.current().state.deployment.phase === 'paused');
    assert.equal(gate.status().ready, false);
    assert.equal(gate.status().startedBy, 'on-demand', 'the paused deployment is still the gate\'s own');
});

test('on demand needs the runner installed: a missing runner gives a refusal and no run', async (t) => {
    const { host, counter } = await startHost(t, { installed: false });
    const verdict = await createLocalLlmGate({ host, settings: onDemand(), pollMs: 10 }).ensure(ENTRY);
    assert.equal(verdict.ok, false);
    assert.match(verdict.reason, /not installed/);
    assert.equal(counter.starts, 0);
});

test('on demand: the idle stop acts only on the gate\'s own deployment, and only when nothing is running or waiting', async (t) => {
    const { host } = await startHost(t);
    let clock = Date.now();
    let busy = true;
    const gate = createLocalLlmGate({ host, settings: onDemand({ idleStopMs: 1000 }), busy: () => busy, now: () => clock, pollMs: 10 });
    assert.equal((await gate.ensure(ENTRY)).ok, true);
    const controller = host.current();
    const pid = controller.state.deployment.runner.pid;
    assert.equal(gate.status().startedBy, 'on-demand');
    // Not idle long enough: nothing is stopped.
    clock += 500;
    assert.equal((await gate.idleCheck()).stopped, false);
    // Idle long enough but busy: still running.
    clock += 5000;
    assert.deepEqual(await gate.idleCheck(), { stopped: false, reason: 'busy' });
    assert.equal(controller.state.deployment.phase, 'ready');
    assert.ok(alive(pid));
    // Idle and not busy: stopped.
    busy = false;
    clock += 5000;
    assert.deepEqual(await gate.idleCheck(), { stopped: true });
    await until(() => !alive(pid));
    assert.equal(controller.state.deployment.phase, 'idle');
    assert.equal(gate.status().startedBy, null);
});

test('the engine\'s cache identity names the model, the runner, its version and the weights\' sha256, and is unknown without them', async (t) => {
    const { host } = await startHost(t);
    const gate = createLocalLlmGate({ host, settings: {} });
    assert.equal(gate.identity(MODEL), null, 'no runner is named and none serves: nothing to identify, so nothing is cached');
    const sha = JSON.parse(fs.readFileSync(SEED_CATALOG_FILE, 'utf8')).models.find((model) => model.id === MODEL).sources.gguf.sha256;
    const identity = gate.identity(MODEL, 'llama.cpp-cpu');
    assert.equal(identity, `local-llm:${MODEL}:llama.cpp-cpu:b11295:${sha}`);
    assert.notEqual(gate.identity(MODEL, 'llama.cpp'), null);
    assert.equal(gate.identity(MODEL, 'fake-gguf'), null, 'a runner the host lock does not pin has no version');
    assert.equal(gate.identity('no-such-model', 'llama.cpp-cpu'), null);
    // The same model with other weights is another identity.
    const catalog = JSON.parse(fs.readFileSync(SEED_CATALOG_FILE, 'utf8'));
    catalog.models.find((model) => model.id === MODEL).sources.gguf.sha256 = 'a'.repeat(64);
    const changed = path.join(scratch(t), 'models.json');
    fs.writeFileSync(changed, JSON.stringify(catalog));
    const other = createLocalLlmGate({ host, settings: {}, seedCatalogFile: changed }).identity(MODEL, 'llama.cpp-cpu');
    assert.equal(other, `local-llm:${MODEL}:llama.cpp-cpu:b11295:${'a'.repeat(64)}`);
    assert.notEqual(other, identity);
    // While a deployment serves the model, its runner is the one named.
    const controller = await host.controller();
    await runModel(controller);
    assert.equal(gate.identity(MODEL), null, 'the stand-in runner has no pinned version');
});

test('the default runner is the CUDA build on an NVIDIA profile and the CPU build otherwise', () => {
    assert.equal(defaultRunnerFor('dedicated'), 'llama.cpp');
    assert.equal(defaultRunnerFor('unified'), 'llama.cpp');
    assert.equal(defaultRunnerFor('cpu'), 'llama.cpp-cpu');
    assert.equal(defaultRunnerFor(null), 'llama.cpp-cpu');
});
