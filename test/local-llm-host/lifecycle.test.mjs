// pworker serve and stop with the local-llm host: default-config serve works with the engine idle, close() drains the engine
// before it returns, a second process for the same home does not run, a port taken by another program is told apart from
// another pworker, and stopProxy waits for the process to exit.
import '../home.mjs';
import assert from 'node:assert/strict';
import { spawn, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import http from 'node:http';
import net from 'node:net';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import { httpFetch } from '../../lib/http-fetch.mjs';
import { createPworkerServer, identifyListener, serve } from '../../lib/server.mjs';
import { proxyPid, stopProxy } from '../../lib/pworker/admin.mjs';
import { readRecords } from '../../lib/local-llm-host/launch.mjs';
import { fakeAdapter, freePort, runModel, scratch, SNAPSHOT, until } from './fixtures/engine.mjs';

const CONTENDER = fileURLToPath(new URL('./fixtures/contender.mjs', import.meta.url));
const SLOW_PROXY = fileURLToPath(new URL('./fixtures/slow-proxy.mjs', import.meta.url));
const HOME_JS = fileURLToPath(new URL('../home.mjs', import.meta.url));

/** The seams that make the host of an in-process server run as on Linux, with the stand-in runner. */
const linuxHost = async (home, extra = {}) => ({
    platformName: 'linux', procDir: path.join(home, 'proc'), ownerPort: await freePort(),
    controllerOptions: {
        runners: { 'fake-gguf': fakeAdapter() }, snapshot: async () => structuredClone(SNAPSHOT),
        download: async ({ artifact }) => ({ status: 'complete', path: `/models/${artifact.file}`, bytesTransferred: 0 }),
        inspect: async () => ({ state: 'complete', bytes: 1 }), remove: async () => 1,
        detectRunner: () => ({ installed: true, version: 'v1', reason: null }), pollMs: 5, stopGraceMs: 500, profile: 'dedicated',
    },
    ...extra,
});

const get = async (base, route) => { const res = await httpFetch(`${base}${route}`); return { status: res.status, body: await res.json().catch(() => null) }; };

test('default-config serve works with the engine idle: /health, /v1/local, the model listing and a clean close', async (t) => {
    const home = scratch(t);
    fs.mkdirSync(path.join(home, 'proc'));
    const env = { PWORKER_HOME: home };
    const logs = [];
    const server = await serve({ port: 0, env, project: false, log: (line) => logs.push(line), localLlmOptions: await linuxHost(home) });
    t.after(() => server.close());
    assert.ok(server.localLlm.describe().ownership.port, 'the home is owned from the start of serve');
    assert.equal(server.localLlm.current(), null, 'and the controller is not created until it is used');
    const base = `http://127.0.0.1:${server.address().port}`;
    const health = await get(base, '/health');
    assert.equal(health.status, 200);
    assert.equal(health.body.ok, true);
    assert.equal((await get(base, '/v1/local')).status, 200);
    assert.equal((await get(base, '/v1/models')).status, 200);
    await server.close();
    assert.equal(server.localLlm.describe().ownership.state, 'closed');
});

test('close() drains the engine before it returns: no runner, no record, no socket, and the owner port is free', async (t) => {
    const home = scratch(t);
    fs.mkdirSync(path.join(home, 'proc'));
    const options = await linuxHost(home);
    const server = await createPworkerServer({ config: { upstreams: {}, defaultUpstream: 'none', tiers: {} }, env: { PWORKER_HOME: home }, localLlmOptions: options });
    await server.localLlm.acquire();
    const controller = await server.localLlm.controller();
    await runModel(controller);
    const target = controller.chatTarget();
    const pid = controller.state.deployment.runner.pid;
    const launchDir = server.localLlm.layout.launchDir;
    assert.equal(readRecords(launchDir).length, 1);
    assert.doesNotThrow(() => process.kill(pid, 0));
    await server.close();
    assert.throws(() => process.kill(pid, 0), { code: 'ESRCH' }, 'the runner process is gone when close() has returned');
    assert.equal(fs.existsSync(target.socketPath), false);
    assert.deepEqual(readRecords(launchDir), []);
    // The owner port can be bound at once: the ownership was released by the drain.
    await new Promise((resolve, reject) => { const probe = net.createServer(); probe.once('error', reject); probe.listen(options.ownerPort, '127.0.0.1', () => probe.close(resolve)); });
    await assert.rejects(server.localLlm.controller(), { code: 'shutting_down' });
});

test('a second pworker for a home whose engine another process owns does not run, after waiting for a predecessor to leave', async (t) => {
    const home = scratch(t);
    fs.mkdirSync(path.join(home, 'proc'));
    // A process that holds the home's owner port (the same port the server is given). freePort() only names a port that was free a
    // moment ago, and the test files run side by side, so another file may bind it first: the holder then reports `refused`, and
    // the test picks another port instead of going on with a home nobody holds.
    let holder = null, ownerPort = null;
    for (let attempt = 0; attempt < 5 && !holder; attempt++) {
        const port = await freePort();
        const child = spawn(process.execPath, [CONTENDER, JSON.stringify({ home, mode: 'bind', holdMs: -1, retryMs: 0, port })], { stdio: ['ignore', 'pipe', 'inherit'] });
        t.after(() => child.kill('SIGKILL'));
        const ev = await new Promise((resolve) => {
            let seen = '';
            child.stdout.on('data', (chunk) => { seen += chunk; if (seen.includes('\n')) resolve(JSON.parse(seen.split('\n')[0]).ev); });
            child.once('exit', () => resolve('exited'));
        });
        if (ev === 'acquired') [holder, ownerPort] = [child, port];
    }
    assert.ok(holder, 'a holder took the owner port');
    const options = await linuxHost(home, { ownerPort });
    const logs = [];
    const started = Date.now();
    const result = await serve({ port: 0, env: { PWORKER_HOME: home }, project: false, log: (line) => logs.push(line), localLlmOptions: options, ownerWaitMs: 500 });
    // A serve that ran after all is closed whatever the assertions find, so a failure here cannot keep the test file alive. It is
    // closed in the test body, not in t.after: the scratch home is removed by an earlier after hook, and close() needs the home.
    try {
        assert.equal(result.already, true);
        assert.ok(Date.now() - started >= 450, 'it waited for the holder to leave');
        assert.equal(fs.existsSync(path.join(home, 'server.json')), false, 'it recorded no server of its own');
        assert.ok(logs.some((line) => /already runs the local models/.test(line)));
    } finally { await result.close?.(); }
});

test('a port held by a program that is not pworker leaves serve running with the engine unavailable, and says so', async (t) => {
    const home = scratch(t);
    fs.mkdirSync(path.join(home, 'proc'));
    const options = await linuxHost(home);
    const holder = net.createServer((socket) => socket.on('error', () => {}));
    await new Promise((resolve) => holder.listen(options.ownerPort, '127.0.0.1', resolve));
    t.after(() => { holder.close(); holder.closeAllConnections?.(); });
    const logs = [];
    const server = await serve({ port: 0, env: { PWORKER_HOME: home }, project: false, log: (line) => logs.push(line), localLlmOptions: options });
    try {
        assert.equal((await get(`http://127.0.0.1:${server.address().port}`, '/health')).status, 200);
        assert.ok(logs.some((line) => /engine is unavailable/.test(line) && /not pworker/.test(line)));
        await assert.rejects(server.localLlm.controller(), { code: 'owner_port_taken' });
    } finally { await server.close(); }
});

test('EADDRINUSE at serve tells another pworker from another program, with distinct results', async (t) => {
    const first = await serve({ port: 0, env: { PWORKER_HOME: scratch(t) }, project: false, log: () => {} });
    try {
        const taken = first.address().port;
        // Another pworker (of another home) on that port.
        const logsB = [];
        const second = await serve({ port: taken, env: { PWORKER_HOME: scratch(t) }, project: false, log: (line) => logsB.push(line) });
        assert.equal(second.already, true);
        assert.equal(second.conflict, undefined);
        assert.ok(logsB.some((line) => /Ploinky Workers proxy already listens/.test(line)));
        assert.equal(await identifyListener('127.0.0.1', taken), 'pworker');
        // Another program: a web server that is not pworker, and one that answers nothing at all.
        for (const behaviour of ['http', 'silent']) {
            const other = http.createServer((req, res) => { if (behaviour === 'http') { res.writeHead(200, { 'content-type': 'application/json' }); res.end('{"ok":true}'); } });
            await new Promise((resolve) => other.listen(0, '127.0.0.1', resolve));
            const port = other.address().port;
            const logs = [];
            const result = await serve({ port, env: { PWORKER_HOME: scratch(t) }, project: false, log: (line) => logs.push(line) });
            other.closeAllConnections();
            other.close();
            assert.equal(result.conflict, true, behaviour);
            assert.equal(result.already, undefined, behaviour);
            assert.ok(logs.some((line) => /taken by another program/.test(line)), behaviour);
            assert.equal(await identifyListener('127.0.0.1', port), 'other');
        }
    } finally { await first.close(); }
});

function startSlowProxy(t, { home, exitMs }) {
    const child = spawn(process.execPath, [SLOW_PROXY], { env: { ...process.env, PWORKER_HOME: home, SLOW_EXIT_MS: String(exitMs) }, stdio: ['ignore', 'pipe', 'inherit'] });
    t.after(() => { try { child.kill('SIGKILL'); } catch {} });
    const exited = new Promise((resolve) => child.once('exit', resolve));
    return new Promise((resolve) => child.stdout.once('data', () => resolve({ child, exited })));
}

test('stopProxy waits until the process has exited, not until /health stops answering', async (t) => {
    const home = scratch(t);
    const env = { PWORKER_HOME: home };
    const { child, exited } = await startSlowProxy(t, { home, exitMs: 1500 });
    assert.equal(await proxyPid(env), child.pid);
    const began = Date.now();
    let alive = null;
    // Half way through the drain: /health no longer answers (what the old stopProxy waited for), yet the process is still there.
    setTimeout(async () => {
        alive = { health: await proxyPid(env), exists: (() => { try { process.kill(child.pid, 0); return true; } catch { return false; } })() };
    }, 700);
    assert.equal(await stopProxy({ env }), true);
    const waited = Date.now() - began;
    assert.deepEqual(alive, { health: null, exists: true }, 'the old exit condition (a failed /health) held while the process still ran');
    assert.ok(waited >= 1400, `stopProxy returned after ${waited} ms, before the process exited`);
    assert.throws(() => process.kill(child.pid, 0), { code: 'ESRCH' }, 'the process is gone when stopProxy has returned');
    await exited;
});

test('stopProxy is bounded: a proxy that does not exit is reported as not stopped', async (t) => {
    const home = scratch(t);
    const env = { PWORKER_HOME: home };
    const { child, exited } = await startSlowProxy(t, { home, exitMs: 'never' });
    const began = Date.now();
    assert.equal(await stopProxy({ env, waitMs: 600 }), false);
    assert.ok(Date.now() - began >= 600);
    assert.doesNotThrow(() => process.kill(child.pid, 0));
    child.kill('SIGKILL');
    await exited;
    assert.equal(await stopProxy({ env }), false, 'no proxy, nothing to stop');
});

test('test/home.mjs removes HF_TOKEN from the environment, with a control showing the token is otherwise inherited', () => {
    const probe = (imports) => spawnSync(process.execPath, [...imports, '-e', 'process.stdout.write(process.env.HF_TOKEN ?? "unset")'], { env: { ...process.env, HF_TOKEN: 'hf_synthetic_probe' }, encoding: 'utf8' }).stdout;
    assert.equal(probe([]), 'hf_synthetic_probe');
    assert.equal(probe(['--import', HOME_JS]), 'unset');
});
