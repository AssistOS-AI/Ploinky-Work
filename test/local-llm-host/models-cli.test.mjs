// `pworker models ...` against a real proxy server in a scratch home: the engine verbs, the progress they print, the exit codes, the
// refusal that `status` shows when the engine is unavailable, the real binary with a recorded server, and `models start|stop NAME`
// that still reach the local servers' routes.
import '../home.mjs';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import http from 'node:http';
import net from 'node:net';
import path from 'node:path';
import { PassThrough } from 'node:stream';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import { createPworkerClient } from '../../lib/client.mjs';
import { isEngineCommand, parseVerbArguments, progressLine, runEngineVerb, runParameters } from '../../lib/pworker/models-cli.mjs';
import { serve } from '../../lib/server.mjs';
import { MODEL, scratch } from './fixtures/engine.mjs';
import { hostSeams, ownedHome, startProxy } from './fixtures/proxy.mjs';

const BIN = fileURLToPath(new URL('../../bin/pworker.mjs', import.meta.url));

function cliIo(extra = {}) {
    const out = [], err = [];
    return { out, err, io: { out: (text) => out.push(text), err: (text) => err.push(text), isTTY: false, home: scratch({ after() {} }), stdin: new PassThrough(), stdout: new PassThrough(), ...extra } };
}
const clientOf = (base) => createPworkerClient({ url: base, purpose: 'test:models-cli', autostart: false, env: {} });
const run = (proxy, args, extra = {}) => { const c = cliIo(extra); return runEngineVerb({ args: ['models', ...args], client: clientOf(proxy.base), io: c.io, pollMs: 20, lineMs: 0 }).then((code) => ({ code, out: c.out.join(''), err: c.err.join('') })); };

test('which commands are the engine\'s: models stop without a name, and every verb; models start NAME and models stop NAME are not', () => {
    assert.equal(isEngineCommand(['models', 'stop']), true);
    assert.equal(isEngineCommand(['models', 'stop', '--json']), true);
    assert.equal(isEngineCommand(['models', 'stop', 'local']), false);
    assert.equal(isEngineCommand(['models', 'start', 'local']), false);
    assert.equal(isEngineCommand(['models']), false);
    assert.equal(isEngineCommand(['models', '--all']), false);
    for (const verb of ['catalog', 'runners', 'lookup', 'add', 'update', 'remove', 'preview', 'run', 'status', 'cancel', 'delete-weights', 'runner', 'hf-token']) assert.equal(isEngineCommand(['models', verb]), true, verb);
    assert.equal(isEngineCommand(['run', 'models']), false);
});

test('flags: --ctx and --parallel are typed, --param k=v repeats with numbers, booleans and null typed, and a bad one is an error', () => {
    const parsed = parseVerbArguments(['qwen', '--runner', 'llama.cpp', '--ctx', '4096', '--param', 'threads=4', '--param', 'fa=true', '--param', 'x=null', '--detach']);
    assert.deepEqual(parsed.positional, ['qwen']);
    assert.equal(parsed.flags.runner, 'llama.cpp');
    assert.equal(parsed.flags.detach, true);
    assert.deepEqual(runParameters(parsed), { ctxSize: 4096, threads: 4, fa: true, x: null });
    assert.equal(runParameters(parseVerbArguments(['m'])), undefined);
    assert.throws(() => runParameters(parseVerbArguments(['m', '--param', 'novalue'])), /key=value/);
    assert.throws(() => runParameters(parseVerbArguments(['m', '--ctx', 'big'])), /needs a number/);
    assert.throws(() => parseVerbArguments(['m', '--runner']), /needs a value/);
});

test('the progress line shows the phase, the bytes of the download, the rate and the time left', () => {
    const status = { deployment: { phase: 'downloading', modelId: 'm1', runnerId: 'llama.cpp', download: { bytes: 512 * 1024 * 1024, total: 1024 * 1024 * 1024, rate: 10 * 1024 * 1024, etaSeconds: 51 } } };
    assert.equal(progressLine(status), 'downloading m1 on llama.cpp: 512 MiB / 1.0 GiB (50%), 10.0 MiB/s, ETA 51s');
    assert.equal(progressLine({ phase: 'idle', deployment: null }), 'phase idle');
    assert.equal(progressLine({ deployment: { phase: 'error', modelId: 'm1', error: 'boom' } }), 'error m1: boom');
});

test('models status --json exits 0 with the engine\'s status, and the text form says what is running', async (t) => {
    const proxy = await startProxy(t);
    const json = await run(proxy, ['status', '--json']);
    assert.equal(json.code, 0);
    const status = JSON.parse(json.out);
    assert.equal(status.available, true);
    assert.equal(status.phase, 'idle');
    assert.equal((await run(proxy, ['status'])).out.trim(), 'phase idle');
});

test('models run shows the phases until the model is ready (exit 0), then status, stop and a second status', async (t) => {
    const proxy = await startProxy(t);
    const ran = await run(proxy, ['run', MODEL, '--runner', 'fake-gguf']);
    assert.equal(ran.code, 0, ran.err + ran.out);
    assert.match(ran.out, new RegExp(`${MODEL} is ready\\.`));
    assert.match(ran.out, /(starting|downloading|loading|ready) /);
    const status = JSON.parse((await run(proxy, ['status', '--json'])).out);
    assert.equal(status.deployment.phase, 'ready');
    assert.equal(status.startedBy, 'admin');
    assert.equal((await run(proxy, ['stop'])).code, 0);
    assert.equal(JSON.parse((await run(proxy, ['status', '--json'])).out).phase, 'idle');
});

test('models run --detach returns at once; a run that cannot start exits 1; an unknown model is an error', async (t) => {
    const proxy = await startProxy(t);
    const detached = await run(proxy, ['run', MODEL, '--runner', 'fake-gguf', '--detach']);
    assert.equal(detached.code, 0);
    assert.match(detached.out, /accepted/);
    await run(proxy, ['stop']);
    await assert.rejects(run(proxy, ['run', 'no-such-model', '--runner', 'fake-gguf']), /No model 'no-such-model'/);
});

test('catalog, runners, preview and delete-weights print what the engine knows', async (t) => {
    const proxy = await startProxy(t);
    const catalog = await run(proxy, ['catalog']);
    assert.match(catalog.out, new RegExp(`ID +NAME[\\s\\S]*${MODEL}`));
    const runners = await run(proxy, ['runners']);
    assert.match(runners.out, /fake-gguf +installed/);
    assert.match(runners.out, /vllm +unsupported +.*not offered by the standalone pworker/);
    const json = JSON.parse((await run(proxy, ['runners', '--json'])).out);
    assert.ok(json.unsupportedRunners.some((runner) => runner.id === 'vllm'));
    const preview = await run(proxy, ['preview', MODEL, '--runner', 'fake-gguf', '--ctx', '2048']);
    assert.match(preview.out, new RegExp(`^${MODEL}: (ok|[a-z-]+)`));
    await assert.rejects(run(proxy, ['preview']), /Usage: pworker models preview/);
});

test('status says why the engine is unavailable when another program holds the home\'s owner port, and models status still exits 0', async (t) => {
    const { home, later } = ownedHome(t);
    const seams = await hostSeams(home);
    const holder = net.createServer((socket) => socket.on('error', () => {}));
    await new Promise((resolve) => holder.listen(seams.ownerPort, '127.0.0.1', resolve));
    later(() => { holder.close(); holder.closeAllConnections?.(); });
    const logs = [];
    const server = await serve({ port: 0, env: { PWORKER_HOME: home }, project: false, log: (line) => logs.push(line), localLlmOptions: seams });
    later(() => server.close());
    const proxy = { base: `http://127.0.0.1:${server.address().port}` };
    const text = await run(proxy, ['status']);
    assert.equal(text.code, 0);
    assert.match(text.out, /engine is unavailable: owner_port_taken/);
    assert.match(text.out, /not pworker/);
    const json = JSON.parse((await run(proxy, ['status', '--json'])).out);
    assert.equal(json.available, false);
    assert.equal(json.refusal.code, 'owner_port_taken');
    // The same refusal is in /v1/local, which is where every client of the proxy looks.
    const local = await (await fetch(`${proxy.base}/v1/local`)).json();
    assert.equal(local['local-llm'].ownership.refusal.code, 'owner_port_taken');
    assert.equal(local['local-llm'].ready, false);
    const failed = await run(proxy, ['run', MODEL, '--runner', 'fake-gguf']).catch((error) => error);
    assert.match(failed.message, /owner_port_taken|not pworker|Port \d+/);
});

// --- the real binary ---

function recordedServer(t, handler) {
    const home = scratch(t);
    const server = http.createServer(handler);
    return new Promise((resolve) => server.listen(0, '127.0.0.1', () => {
        t.after(() => { server.closeAllConnections(); server.close(); });
        // The recorded server of the home: the process that is running these tests (alive), at the stand-in's port.
        fs.writeFileSync(path.join(home, 'server.json'), `${JSON.stringify({ pid: process.pid, host: '127.0.0.1', port: server.address().port })}\n`);
        resolve({ home, port: server.address().port });
    }));
}

function bin(home, args, { input = null, env = {} } = {}) {
    const { NODE_TEST_CONTEXT: _context, ...parent } = process.env;
    const child = spawn(process.execPath, [BIN, ...args], { env: { ...parent, PWORKER_HOME: home, ...env }, stdio: ['pipe', 'pipe', 'pipe'] });
    let stdout = '', stderr = '';
    child.stdout.on('data', (chunk) => { stdout += chunk; });
    child.stderr.on('data', (chunk) => { stderr += chunk; });
    child.stdin.end(input ?? '');
    return new Promise((resolve) => child.on('close', (code) => resolve({ code, stdout, stderr })));
}

test('the binary: models status --json against a running proxy exits 0 and prints its status', async (t) => {
    const proxy = await startProxy(t);
    const home = scratch(t);
    fs.writeFileSync(path.join(home, 'server.json'), `${JSON.stringify({ pid: process.pid, host: '127.0.0.1', port: Number(new URL(proxy.base).port) })}\n`);
    const result = await bin(home, ['models', 'status', '--json']);
    assert.equal(result.code, 0, result.stderr);
    assert.equal(JSON.parse(result.stdout).phase, 'idle');
    const runners = await bin(home, ['models', 'runners']);
    assert.equal(runners.code, 0, runners.stderr);
    assert.match(runners.stdout, /fake-gguf/);
});

test('the binary: models stop NAME still posts to /v1/local/NAME/stop, models stop alone goes to the engine', async (t) => {
    const requests = [];
    const { home } = await recordedServer(t, (req, res) => {
        requests.push(`${req.method} ${req.url}`);
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(req.url === '/health' ? JSON.stringify({ ok: true, uptime_s: 1, tiers: [], upstreams: {} }) : JSON.stringify({ ok: true, deployment: null, status: {} }));
    });
    const named = await bin(home, ['models', 'stop', 'localnano']);
    assert.equal(named.code, 0, named.stderr);
    assert.ok(requests.includes('POST /v1/local/localnano/stop'), requests.join(', '));
    assert.ok(!requests.includes('POST /v1/local-llm/stop'));
    requests.length = 0;
    const engine = await bin(home, ['models', 'stop']);
    assert.equal(engine.code, 0, engine.stderr);
    assert.ok(requests.includes('POST /v1/local-llm/stop'), requests.join(', '));
    requests.length = 0;
    await bin(home, ['models', 'start', 'localnano']);
    assert.ok(requests.includes('POST /v1/local/localnano/start'), requests.join(', '));
});

test('models run and preview without --runner send no runnerId at all (never the string "undefined")', async () => {
    const seen = [];
    const client = { engine: { run: async (body) => { seen.push(['run', body]); return { accepted: true, deployment: { phase: 'ready', modelId: body.modelId } }; }, preview: async (body) => { seen.push(['preview', body]); return { preview: {} }; }, overview: async () => ({ runners: [] }), status: async () => ({ deployment: { phase: 'ready', modelId: 'm' } }) } };
    const c = cliIo();
    await runEngineVerb({ args: ['models', 'run', 'some-model', '--detach'], client, io: c.io, pollMs: 5, lineMs: 0 });
    await runEngineVerb({ args: ['models', 'preview', 'some-model'], client, io: cliIo().io, pollMs: 5, lineMs: 0 });
    assert.deepEqual(seen.map(([name]) => name), ['run', 'preview']);
    for (const [, body] of seen) assert.equal(JSON.stringify(body).includes('runnerId'), false, JSON.stringify(body));
});
