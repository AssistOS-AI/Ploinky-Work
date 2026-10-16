// R9, the engine's Unix-socket seam: `host.listen` makes a llama.cpp start listen on <socketDir>/<8 hex>.sock, probe it over
// the socket and hand out a socket chat target, while a host without `listen` keeps the TCP launch, probes, state and chat
// target it always had (byte for byte, compared as JSON so key order counts).
import '../home.mjs';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { loadSeedCatalog } from '../../lib/local-llm/controller/catalog.mjs';
import { createStateStore } from '../../lib/local-llm/controller/stateStore.mjs';
import { defineHost } from '../../lib/local-llm/host.mjs';
import { SEED_CATALOG_FILE } from '../local-llm/fixtures/seedCatalog.mjs';
import { agentLikeHost, createController } from '../local-llm/fixtures/host.mjs';

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;
const SEED = loadSeedCatalog(SEED_CATALOG_FILE);
const KEY = 'k'.repeat(43);
const SNAPSHOT = Object.freeze({
    gpu: { available: true, name: 'Test GPU', driverVersion: '595.91.07', totalBytes: 6144 * MIB, usedBytes: 144 * MIB, freeBytes: 6000 * MIB, processes: [] },
    memory: { totalBytes: 31 * GIB, availableBytes: 24 * GIB },
    disk: { freeBytes: 300 * GIB, totalBytes: 500 * GIB },
    cpus: 20,
});

function harness(t, { listen = null, download, startIdFactory } = {}) {
    const dataDir = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), 'll-sock-'));
    t.after(() => fs.rmSync(dataDir, { recursive: true, force: true }));
    const started = [];
    const probes = [];
    const downloads = [];
    const host = agentLikeHost({ dataDir, ...(listen ? { listen } : {}) });
    const controller = createController({
        host,
        dataDir,
        env: { PATH: '/usr/bin' },
        seedCatalog: SEED,
        stateStore: createStateStore({ dataDir }),
        snapshot: async () => structuredClone(SNAPSHOT),
        download: download || (async () => { downloads.push(1); return { status: 'complete', path: '/data/models/gguf/gpt.gguf', bytesTransferred: 0 }; }),
        inspect: async () => ({ state: 'absent', bytes: 0 }),
        remove: async () => 0,
        startRunner(options) {
            let running = true;
            let resolveExit;
            const handle = {
                pid: 5000 + started.length, ...options,
                exited: new Promise((resolve) => { resolveExit = resolve; }),
                get running() { return running; },
                async stop() { running = false; resolveExit({ code: 0, signal: 'SIGTERM', error: null }); return handle.exited; },
            };
            started.push(handle);
            return handle;
        },
        fetchImpl: async (url, options = {}) => {
            probes.push(['tcp', String(url), options.headers?.authorization ?? null]);
            return { ok: true, status: 200, json: async () => ({}) };
        },
        socketFetchImpl: async (url, options = {}) => {
            probes.push(['socket', String(url), options.headers?.authorization ?? null, options.socketPath]);
            return { ok: true, status: 200, json: async () => ({}) };
        },
        ...(startIdFactory ? { startIdFactory } : {}),
        apiKeyFactory: () => KEY,
        detectRunner: (runner) => ({ installed: runner.supported, version: runner.pinnedVersion, reason: null }),
        pollMs: 2,
        stopGraceMs: 50,
    });
    return { controller, started, probes, downloads, dataDir };
}

async function until(predicate, timeoutMs = 2000) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        if (await predicate()) return;
        await new Promise((resolve) => setTimeout(resolve, 5));
    }
    assert.fail('condition not reached');
}

test('defineHost: listen is null by default and validated when given', () => {
    const base = { dataDir: '/d', runRoot: '/r' };
    assert.equal(defineHost(base).listen, null);
    const host = defineHost({ ...base, listen: { socketDir: '/s/r', pathHint: 'shorter please' } });
    assert.deepEqual(host.listen, { socketDir: '/s/r', pathHint: 'shorter please' });
    assert.equal(Object.isFrozen(host.listen), true);
    assert.equal(defineHost({ ...base, listen: { socketDir: '/s/r' } }).listen.pathHint, null);
    for (const [listen, field] of [
        [{ socketDir: 'relative/r' }, 'listen.socketDir'], [{ socketDir: '/a:b' }, 'listen.socketDir'], [{ socketDir: '/a,b' }, 'listen.socketDir'],
        [{ socketDir: '/a/../b' }, 'listen.socketDir'], [{}, 'listen.socketDir'], [{ socketDir: '/s', extra: 1 }, 'listen.extra'],
        [{ socketDir: '/s', pathHint: '' }, 'listen.pathHint'], ['/s', 'listen'],
    ]) {
        assert.throws(() => defineHost({ ...base, listen }), (error) => error.code === 'invalid_host' && error.details?.field === field, JSON.stringify(listen));
    }
});

test('a host without listen keeps the TCP launch, probes, state and chat target byte for byte', async (t) => {
    const h = harness(t);
    await h.controller.run({ modelId: 'gpt-oss-20b', runnerId: 'llama.cpp', requestId: 'request-tcp-01' });
    await until(() => h.controller.state.deployment?.phase === 'ready');
    const [process] = h.started;
    assert.equal(JSON.stringify(process.args.slice(0, 8)), JSON.stringify(['-m', '/data/models/gguf/gpt.gguf', '--host', '127.0.0.1', '--port', '18080', '--api-key', KEY]));
    assert.equal('listen' in process, false, 'a TCP start passes no listen option to the process starter');
    assert.equal(JSON.stringify(h.probes), JSON.stringify([
        ['tcp', 'http://127.0.0.1:18080/health', null],
        ['tcp', 'http://127.0.0.1:18080/v1/models', `Bearer ${KEY}`],
    ]));
    assert.equal(JSON.stringify(h.controller.state.deployment.runner), JSON.stringify({
        pid: 5000, port: 18080, startedAt: h.controller.state.deployment.runner.startedAt,
    }));
    assert.equal(JSON.stringify(h.controller.chatTarget()), JSON.stringify({
        runnerId: 'llama.cpp', modelId: 'gpt-oss-20b', baseUrl: 'http://127.0.0.1:18080',
        apiKey: KEY, model: 'gpt-oss-20b', requestOptions: null,
    }));
    await h.controller.stop();
});

test('a socket host starts llama.cpp on <socketDir>/<8 hex>.sock with no TCP port and probes over the socket', async (t) => {
    const socketDir = '/s/r';
    const h = harness(t, { listen: { socketDir }, startIdFactory: () => 'a1b2c3d4' });
    await h.controller.run({ modelId: 'gpt-oss-20b', runnerId: 'llama.cpp', requestId: 'request-sock-01' });
    await until(() => h.controller.state.deployment?.phase === 'ready');
    const [process] = h.started;
    const socketPath = `${socketDir}/a1b2c3d4.sock`;
    assert.deepEqual(process.listen, { startId: 'a1b2c3d4', socketPath });
    assert.deepEqual(process.args.slice(0, 5), ['-m', '/data/models/gguf/gpt.gguf', '--host', socketPath, '--api-key']);
    assert.equal(process.args.includes('--port'), false);
    assert.deepEqual(h.probes, [
        ['socket', 'http://localhost/health', null, socketPath],
        ['socket', 'http://localhost/v1/models', `Bearer ${KEY}`, socketPath],
    ]);
    assert.equal(h.controller.state.deployment.runner.port, null);
    assert.equal(h.controller.state.deployment.runner.socketPath, socketPath);
    assert.deepEqual(h.controller.chatTarget(), {
        runnerId: 'llama.cpp', modelId: 'gpt-oss-20b', baseUrl: 'http://localhost', socketPath,
        apiKey: KEY, model: 'gpt-oss-20b', requestOptions: null,
    });
    await h.controller.stop();
});

test('every start gets its own 8-hex socket name', async (t) => {
    const h = harness(t, { listen: { socketDir: '/s/r' } });
    const names = [];
    for (const id of ['request-sock-11', 'request-sock-12']) {
        await h.controller.run({ modelId: 'gpt-oss-20b', runnerId: 'llama.cpp', requestId: id, replace: true });
        await until(() => h.controller.state.deployment?.phase === 'ready' && h.started.length === names.length + 1);
        names.push(h.started.at(-1).listen.socketPath);
    }
    for (const name of names) assert.match(path.basename(name), /^[0-9a-f]{8}\.sock$/);
    assert.notEqual(names[0], names[1]);
    await h.controller.stop();
});

test('a socket path that cannot fit sun_path fails the run before anything is downloaded or launched, naming the limit and both fixes', async (t) => {
    const socketDir = `/${'d'.repeat(120)}`;
    const h = harness(t, { listen: { socketDir, pathHint: 'Use a shorter PWORKER_HOME, or set localLlm.socketDir.' } });
    await assert.rejects(
        h.controller.run({ modelId: 'gpt-oss-20b', runnerId: 'llama.cpp', requestId: 'request-long-01' }),
        (error) => {
            assert.equal(error.code, 'socket_path_too_long');
            assert.equal(error.details.limit, 107);
            assert.match(error.message, /107 bytes/);
            assert.match(error.message, /PWORKER_HOME/);
            assert.match(error.message, /localLlm\.socketDir/);
            return true;
        });
    assert.equal(h.downloads.length, 0, 'nothing was downloaded');
    assert.equal(h.started.length, 0, 'nothing was launched');
});

test('the socket limit is the platform\'s: a platform with a shorter sun_path refuses a path Linux accepts', async (t) => {
    const dataDir = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), 'll-sock-'));
    t.after(() => fs.rmSync(dataDir, { recursive: true, force: true }));
    const socketDir = `/${'d'.repeat(90)}`;
    const probe = `${socketDir}/00000000.sock`;
    assert.ok(Buffer.byteLength(probe) > 103 && Buffer.byteLength(probe) <= 107);
    const { linuxPlatform } = await import('../../lib/local-llm/host.mjs');
    const platform = Object.freeze({ ...linuxPlatform(), socketPathMax: 103 });
    const host = agentLikeHost({ dataDir, listen: { socketDir }, platform });
    const controller = createController({
        host, dataDir, env: { PATH: '/usr/bin' }, seedCatalog: SEED, stateStore: createStateStore({ dataDir }),
        snapshot: async () => structuredClone(SNAPSHOT), detectRunner: (runner) => ({ installed: true, version: runner.pinnedVersion, reason: null }),
    });
    await assert.rejects(controller.run({ modelId: 'gpt-oss-20b', runnerId: 'llama.cpp', requestId: 'request-long-02' }), (error) => error.code === 'socket_path_too_long' && error.details.limit === 103);
});

test('a runner that cannot listen on a Unix socket is refused on a socket host', async (t) => {
    const h = harness(t, { listen: { socketDir: '/s/r' } });
    await h.controller.addModel({ id: 'olla-user', sources: { ollama: { type: 'ollama', tag: 'gpt-oss:20b' } } });
    await assert.rejects(
        h.controller.run({ modelId: 'olla-user', runnerId: 'ollama', requestId: 'request-ollama-01' }),
        (error) => error.code === 'runner_unavailable' && /cannot listen on a Unix socket/.test(error.message));
    assert.equal(h.started.length, 0);
});
