// Fixes that came with the gate and the routes: the version the host's llama.cpp reports, the home a server gives its host, a drain
// that cancels a pending acquisition, how long a starting proxy is waited for, and what `pworker stop` can tell apart.
import '../home.mjs';
import assert from 'node:assert/strict';
import { spawn, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import { AUTOSTART_WAIT_MS } from '../../lib/client.mjs';
import { createLocalLlmHost, OWNER_WAIT_MS } from '../../lib/local-llm-host/host.mjs';
import { acquireOwnership } from '../../lib/local-llm-host/ownership.mjs';
import { defineHost } from '../../lib/local-llm/host.mjs';
import { createRunners } from '../../lib/local-llm/runners/index.mjs';
import { proxyPid, startProxy, START_WAIT_MS, stopProxyResult } from '../../lib/pworker/admin.mjs';
import { freePort, scratch } from './fixtures/engine.mjs';
import { hostSeams } from './fixtures/proxy.mjs';

const SLOW_PROXY = fileURLToPath(new URL('./fixtures/slow-proxy.mjs', import.meta.url));

test('the host\'s llama.cpp reports the version its lock installs (b11295); the agent\'s keeps its own (b11159)', async (t) => {
    const agent = createRunners({ host: defineHost({ dataDir: '/x/data', runRoot: '/x/run' }) });
    assert.equal(agent['llama.cpp'].pinnedVersion, 'b11159', 'the agent path is unchanged');
    for (const [label, which, profile, runnerId] of [['NVIDIA', () => '/usr/bin/nvidia-smi', 'dedicated', 'llama.cpp'], ['CPU', () => null, 'cpu', 'llama.cpp-cpu']]) {
        const home = scratch(t);
        const seams = await hostSeams(home, { extra: { detectRunner: (runner) => ({ installed: false, version: null, reason: runner.pinnedVersion }) } });
        delete seams.controllerOptions.runners;
        seams.controllerOptions.profile = profile;
        const host = createLocalLlmHost({ home, env: { PWORKER_HOME: home, PATH: '/usr/bin' }, which, ...seams });
        t.after(() => host.drain());
        await host.acquire();
        const overview = await (await host.controller()).overview();
        const runner = overview.runners.find((entry) => entry.id === runnerId);
        assert.ok(runner, `${label}: ${runnerId} is offered`);
        assert.equal(runner.pinnedVersion, 'b11295', label);
    }
});

test('drain cancels an acquisition that is still waiting for the home: nothing is acquired or reaped after the shutdown request', async (t) => {
    const home = scratch(t);
    const seams = await hostSeams(home);
    const kills = [];
    const holder = await acquireOwnership({ realHome: fs.realpathSync(home), port: seams.ownerPort, pid: 4242 });
    t.after(() => holder.release());
    const host = createLocalLlmHost({ home, env: { PWORKER_HOME: home }, killImpl: (...args) => kills.push(args), ...seams });
    const pending = host.acquire({ waitMs: 30_000 });
    const outcome = pending.then(() => 'acquired', (error) => error.code);
    await new Promise((resolve) => setTimeout(resolve, 400));
    const began = Date.now();
    await host.drain();
    assert.ok(Date.now() - began < 3000, 'the drain did not wait for the 30 s window');
    assert.equal(await outcome, 'shutting_down');
    assert.deepEqual(kills, []);
    // The holder still holds the port; the drained host took nothing.
    await assert.rejects(acquireOwnership({ realHome: fs.realpathSync(home), port: seams.ownerPort }), { code: 'owner_held' });
});

test('a starting proxy is waited for at least as long as serve can wait for the previous one, and an autostarting client for longer still', () => {
    assert.ok(START_WAIT_MS >= OWNER_WAIT_MS + 10_000, `${START_WAIT_MS}`);
    assert.ok(AUTOSTART_WAIT_MS >= START_WAIT_MS, `${AUTOSTART_WAIT_MS} >= ${START_WAIT_MS}`);
});

test('startProxy starts a real serve in the home and stopProxyResult then says stopped, and not-running after it', async (t) => {
    const home = scratch(t);
    const env = { ...process.env, PWORKER_HOME: home, PWORKER_PORT: String(await freePort()) };
    t.after(async () => { await stopProxyResult({ env, waitMs: 20_000 }); });
    assert.equal(await startProxy({ env, waitMs: 60_000 }), true);
    const pid = await proxyPid(env);
    assert.ok(pid);
    assert.equal(await startProxy({ env }), false, 'already running');
    assert.deepEqual(await stopProxyResult({ env }), { state: 'stopped', pid });
    assert.deepEqual(await stopProxyResult({ env }), { state: 'not-running', pid: null });
});

test('startProxy fails at once when the process exits without answering because its port is taken by another program', async (t) => {
    const home = scratch(t);
    const other = http.createServer((req, res) => res.end('not pworker'));
    await new Promise((resolve) => other.listen(0, '127.0.0.1', resolve));
    t.after(() => { other.closeAllConnections(); other.close(); });
    const env = { ...process.env, PWORKER_HOME: home, PWORKER_PORT: String(other.address().port) };
    const began = Date.now();
    await assert.rejects(startProxy({ env, waitMs: 60_000 }), /port is taken by another program/);
    assert.ok(Date.now() - began < 20_000, 'it did not wait for the whole window');
});

test('stopProxyResult tells a proxy that is still draining from one that is not running', async (t) => {
    const home = scratch(t);
    const env = { PWORKER_HOME: home };
    const child = spawn(process.execPath, [SLOW_PROXY], { env: { ...process.env, PWORKER_HOME: home, SLOW_EXIT_MS: 'never' }, stdio: ['ignore', 'pipe', 'inherit'] });
    t.after(() => child.kill('SIGKILL'));
    await new Promise((resolve) => child.stdout.once('data', resolve));
    assert.deepEqual(await stopProxyResult({ env, waitMs: 400 }), { state: 'still-running', pid: child.pid });
    child.kill('SIGKILL');
    await new Promise((resolve) => child.once('exit', resolve));
    assert.deepEqual(await stopProxyResult({ env }), { state: 'not-running', pid: null });
});

test('test/home.mjs points HOME at the scratch home, so "~/.pworker" of a fixture without PWORKER_HOME is inside it, with a control', () => {
    const home = fileURLToPath(new URL('../home.mjs', import.meta.url));
    const probe = (imports) => spawnSync(process.execPath, [...imports, '-e', 'process.stdout.write(JSON.stringify({ home: process.env.HOME, pworker: process.env.PWORKER_HOME ?? null }))'], { env: { ...process.env, HOME: '/real/home/of/the/user', PWORKER_HOME: '' }, encoding: 'utf8' }).stdout;
    const control = JSON.parse(probe([]));
    assert.equal(control.home, '/real/home/of/the/user');
    const inside = JSON.parse(probe(['--import', home]));
    assert.equal(inside.home, inside.pworker);
    assert.notEqual(inside.home, '/real/home/of/the/user');
    assert.ok(inside.home.startsWith(fs.realpathSync(os.tmpdir())));
});
