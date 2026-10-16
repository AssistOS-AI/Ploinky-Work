// The proxy with a provider of kind `local-llm`: a refusal falls back like a local server's, a ready deployment is reached over its
// Unix socket with the per-start key, and an idle engine is never created by anything that does not use it. A real proxy server on a
// free port, a real controller, and the stand-in runner (fixtures/fake-runner.mjs) listening on a real socket.
import '../home.mjs';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import test from 'node:test';

import { createCore } from '../../lib/core.mjs';
import { httpFetch } from '../../lib/http-fetch.mjs';
import { MODEL, runModel } from './fixtures/engine.mjs';
import { chat, request, startProxy } from './fixtures/proxy.mjs';

const seen = async (socketPath) => (await httpFetch('http://localhost/seen', { socketPath })).json();

test('a refusal of the engine falls back down the tier chain as an unavailable local server does', async (t) => {
    const proxy = await startProxy(t);
    const answer = await chat(proxy.base, 'tiny');
    assert.equal(answer.status, 200);
    assert.equal(answer.headers['x-pworker-fallback-reason'], 'unavailable');
    assert.equal(answer.headers['x-pworker-model'], 'fb/fb-model');
    assert.equal(answer.body.choices[0].message.content, 'fallback answer');
    const refused = proxy.server.core.monitor.records.find((record) => record.upstream === 'local-llm');
    assert.equal(refused.status, 503);
    assert.equal(refused.not_sent, true);
    assert.match(refused.error, /^local server unavailable: No local model is ready; run one with `pworker models run <id>`/);
    const fellBack = proxy.server.core.monitor.records.find((record) => record.upstream === 'fb');
    assert.equal(fellBack.fallback_from, 'local-llm');
    assert.equal(fellBack.fallback_kind, 'unavailable');
    assert.match(fellBack.fallback_reason, /No local model is ready/);
    assert.equal(proxy.host.current(), null, 'the refusal did not create the engine');
});

test('without a fallback entry the refusal is a 503 local_unavailable', async (t) => {
    const proxy = await startProxy(t);
    const answer = await chat(proxy.base, 'solo');
    assert.equal(answer.status, 503);
    assert.equal(answer.body.error.type, 'local_unavailable');
    assert.match(answer.body.error.message, /No local model is ready/);
    assert.equal(proxy.calls.filter((call) => call.method === 'POST').length, 0, 'nothing was sent to the other provider');
});

test('a ready deployment is reached over its Unix socket with the per-start key, the model is rewritten and the client\'s Authorization is never forwarded', async (t) => {
    const proxy = await startProxy(t);
    const controller = await proxy.host.controller();
    await runModel(controller);
    const target = controller.chatTarget();
    assert.equal(target.baseUrl, 'http://localhost');
    const answer = await chat(proxy.base, 'tiny', {}, { authorization: 'Bearer client-token-that-must-not-travel' });
    assert.equal(answer.status, 200);
    assert.equal(answer.headers['x-pworker-model'], `local-llm/${MODEL}`);
    assert.equal(answer.body.choices[0].message.content, `echo fake/${MODEL}`, 'the runner got the engine\'s model name, not the catalog id');
    const calls = await seen(target.socketPath);
    assert.equal(calls.length, 1);
    assert.deepEqual(calls[0], { model: `fake/${MODEL}`, key: target.apiKey });
    assert.notEqual(calls[0].key, 'client-token-that-must-not-travel');
    assert.equal(proxy.calls.filter((call) => call.method === 'POST').length, 0, 'the fallback was not used');
    const record = proxy.server.core.monitor.records.find((entry) => entry.upstream === 'local-llm' && entry.status === 200);
    assert.equal(record.model, MODEL, 'the request log keeps the catalog id');
    // The key of the run and the Hugging Face token are among the secrets kept out of logs and errors.
    assert.ok(proxy.server.core.monitor.secrets.includes(target.apiKey));
});

test('a streamed answer passes through the socket as it is produced', async (t) => {
    const proxy = await startProxy(t);
    const controller = await proxy.host.controller();
    await runModel(controller);
    const chunks = await new Promise((resolve, reject) => {
        const body = JSON.stringify({ model: 'solo', stream: true, messages: [{ role: 'user', content: 'hi' }] });
        const req = http.request(`${proxy.base}/v1/chat/completions`, { method: 'POST', headers: { 'content-type': 'application/json', 'content-length': Buffer.byteLength(body), 'x-pworker-purpose': 'test:local-llm' } }, (res) => {
            const parts = [];
            res.on('data', (chunk) => parts.push({ at: Date.now(), text: chunk.toString('utf8'), status: res.statusCode, type: res.headers['content-type'] }));
            res.on('end', () => resolve(parts));
        });
        req.on('error', reject);
        req.end(body);
    });
    assert.equal(chunks[0].status, 200);
    assert.match(chunks[0].type, /text\/event-stream/);
    const text = chunks.map((chunk) => chunk.text).join('');
    assert.match(text, /"content":"one "/);
    assert.match(text, /"content":"two"/);
    assert.match(text, /data: \[DONE\]/);
    assert.ok(chunks.length >= 2 && chunks.at(-1).at - chunks[0].at >= 15, 'the events arrived over time, not as one buffered body');
    assert.equal((await seen(controller.chatTarget().socketPath)).at(-1).model, `fake/${MODEL}`);
});

test('/v1/models of the engine is built fresh on every call: a model that became ready shows at once', async (t) => {
    const proxy = await startProxy(t);
    const before = await request(proxy.base, '/u/local-llm/v1/models');
    assert.equal(before.status, 200);
    const row = before.body.data.find((model) => model.id === MODEL);
    assert.equal(row.x_local_llm.ready, false);
    assert.match(row.x_local_llm.sha256, /^[0-9a-f]{64}$/);
    const controller = await proxy.host.controller();
    await runModel(controller);
    const after = await request(proxy.base, '/u/local-llm/v1/models');
    assert.equal(after.body.data.find((model) => model.id === MODEL).x_local_llm.ready, true);
    assert.equal(after.body.data.find((model) => model.id === MODEL).x_local_llm.runnerId, 'fake-gguf');
    await controller.stop();
    assert.equal((await request(proxy.base, '/u/local-llm/v1/models')).body.data.find((model) => model.id === MODEL).x_local_llm.ready, false, 'and it goes at once, too');
});

test('/v1/local shows the engine next to the local servers, without creating it, and models stop NAME reaches its gate', async (t) => {
    const proxy = await startProxy(t);
    const local = await request(proxy.base, '/v1/local');
    assert.equal(local.status, 200);
    const row = local.body['local-llm'];
    assert.equal(row.kind, 'local-llm');
    assert.equal(row.running, false);
    assert.equal(row.ownership.state, 'idle');
    assert.equal(proxy.host.current(), null);
    const started = await request(proxy.base, '/v1/local/local-llm/start', { method: 'POST', body: {} });
    assert.equal(started.status, 404);
    assert.match(started.body.error.message, /pworker models run/);
    const controller = await proxy.host.controller();
    await runModel(controller);
    assert.equal((await request(proxy.base, '/v1/local')).body['local-llm'].running, true);
    const stopped = await request(proxy.base, '/v1/local/local-llm/stop', { method: 'POST', body: {} });
    assert.equal(stopped.status, 200);
    assert.equal(controller.state.deployment.phase, 'idle');
});

test('strict cache with an engine that cannot be identified is a miss, and no model is called', async (t) => {
    const proxy = await startProxy(t, { config: { cache: { defaultMode: 'use' } } });
    const answer = await chat(proxy.base, 'tiny', {}, { 'x-pworker-cache': 'strict' });
    assert.equal(answer.status, 409);
    assert.equal(answer.body.error.type, 'cache_miss');
    assert.equal(proxy.calls.filter((call) => call.method === 'POST').length, 0);
    // A request to another provider keeps its cache: its answer is stored and replayed.
    const first = await request(proxy.base, '/v1/chat/completions', { method: 'POST', body: { model: 'fb-model', messages: [{ role: 'user', content: 'cache me' }] }, headers: { 'x-pworker-cache': 'use' } });
    const second = await request(proxy.base, '/v1/chat/completions', { method: 'POST', body: { model: 'fb-model', messages: [{ role: 'user', content: 'cache me' }] }, headers: { 'x-pworker-cache': 'use' } });
    assert.equal(first.status, 200);
    assert.equal(second.headers['x-pworker-cache'], 'hit');
});

test('nothing that does not use the engine creates it: no directory, no listener, no controller', async (t) => {
    // A host on a Linux platform whose file system and network refuse every change and every listener, and record the attempt.
    const attempts = [];
    const readOnly = new Set(['statSync', 'lstatSync', 'readFileSync', 'realpathSync', 'readdirSync', 'existsSync', 'accessSync', 'readlinkSync', 'fstatSync', 'constants']);
    const fsApi = new Proxy(fs, { get: (target, name) => (typeof target[name] === 'function' && !readOnly.has(name) ? (...args) => { attempts.push(`fs.${String(name)}`); throw new Error(`fs.${String(name)} must not be called by an idle engine`); } : target[name]) });
    const netImpl = { createServer: () => { attempts.push('net.createServer'); throw new Error('no listener'); }, connect: () => { attempts.push('net.connect'); throw new Error('no connection'); } };
    const proxy = await startProxy(t, { hostOptions: { extra: {} }, config: { cache: { defaultMode: 'use' } }, env: {} });
    // startProxy built the server with the ordinary seams; build the guarded one around the same home.
    const { createPworkerServer } = await import('../../lib/server.mjs');
    const guarded = await createPworkerServer({
        config: proxy.server.core.config, env: {}, coreOptions: { fetchImpl: async (url) => (String(url).endsWith('/v1/models') ? Response.json({ data: [] }) : Response.json({ choices: [{ message: { content: 'ok' }, finish_reason: 'stop' }], usage: {} })) },
        localLlmOptions: { platformName: 'linux', fsApi, netImpl, ownerPort: 1 },
    });
    const address = await guarded.listen(0);
    proxy.later(() => guarded.close());
    const base = `http://127.0.0.1:${address.port}`;
    for (const [route, options] of [['/health', {}], ['/v1/local', {}], ['/stats', {}], ['/v1/models', {}], ['/u/local-llm/v1/models', {}], ['/v1/local/local-llm/stop', { method: 'POST', body: {} }], ['/v1/local-llm/stop', { method: 'POST', body: {} }]]) {
        const answer = await request(base, route, options);
        assert.ok(answer.status < 500, `${route}: ${answer.status}`);
    }
    assert.equal((await chat(base, 'fb-model')).status, 200, 'a request to another provider');
    assert.equal((await chat(base, 'tiny')).status, 200, 'a tier whose engine entry is refused falls back');
    assert.equal(guarded.localLlm.current(), null);
    assert.equal(guarded.localLlm.describe().ownership.state, 'idle');
    assert.deepEqual(attempts, [], 'the engine touched neither the file system nor the network');
    // Negative control: the same guards do catch the host when it is acquired.
    await assert.rejects(guarded.localLlm.acquire());
    assert.ok(attempts.some((attempt) => attempt.startsWith('fs.')), 'the guarded file system records the attempt of an acquisition');
});

test('more than one provider of kind local-llm is refused when the core is built', () => {
    const provider = { kind: 'local-llm', baseUrl: 'http://localhost', noKey: true };
    assert.throws(() => createCore({ config: { defaultUpstream: 'a', providers: { a: provider, b: provider } }, env: {}, dataDir: path.join(scratchDir(), 'data') }), /at most one provider of kind "local-llm"/);
    assert.throws(() => createCore({ config: { defaultUpstream: 'a', providers: { a: { ...provider, start: { bin: 'llama-server' } } } }, env: {}, dataDir: path.join(scratchDir(), 'data') }), /cannot have a start block/);
});

function scratchDir() {
    return fs.mkdtempSync(path.join(fs.realpathSync(process.env.TMPDIR || '/tmp'), 'lh-core-'));
}

test('the per-start key and the Hugging Face token are in no response, no request log and no file of the home but the token file (and the engine\'s private state)', async (t) => {
    const token = 'hf_synthetic_token_for_the_leak_scan_0123456789';
    const proxy = await startProxy(t, { env: { HF_TOKEN: token } });
    const controller = await proxy.host.controller();
    await runModel(controller);
    const target = controller.chatTarget();
    assert.equal((await chat(proxy.base, 'tiny')).status, 200);
    assert.equal((await chat(proxy.base, 'tiny', { stream: true })).status, 200);
    const responses = {};
    for (const route of ['/v1/local-llm', '/v1/local-llm/status', '/v1/local', '/stats', '/health', '/v1/models', '/u/local-llm/v1/models', '/v1/ops']) responses[route] = (await request(proxy.base, route)).text;
    // The runner's log lines (what status returns) as the engine kept them, too.
    responses.logs = JSON.stringify(controller.state.deployment?.logs ?? []);
    const secrets = { key: target.apiKey, token };
    for (const [route, text] of Object.entries(responses)) {
        for (const [name, secret] of Object.entries(secrets)) assert.ok(!text.includes(secret), `${name} is in the answer of ${route}`);
    }
    const holders = (secret) => {
        const hits = [];
        const walk = (dir) => {
            for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
                const full = path.join(dir, entry.name);
                if (entry.isDirectory()) walk(full);
                else if (entry.isFile() && fs.readFileSync(full).includes(secret)) hits.push(path.relative(proxy.home, full));
            }
        };
        walk(proxy.home);
        return hits;
    };
    assert.deepEqual(holders(token), [], 'the token came from the environment and is written nowhere');
    // The key is the engine's own and may sit in its private state (mode 0600); it is in no log, no data file and no other place.
    const keyHolders = holders(target.apiKey);
    assert.ok(keyHolders.every((file) => file.startsWith(path.join('local-llm', 'state'))), `the key is in ${keyHolders.join(', ')}`);
    for (const file of keyHolders) assert.equal(fs.statSync(path.join(proxy.home, file)).mode & 0o077, 0, `${file} is private`);
    // Negative control: the scan does see a planted copy in a request log.
    fs.mkdirSync(path.join(proxy.home, 'data'), { recursive: true });
    fs.writeFileSync(path.join(proxy.home, 'data', 'requests-planted.jsonl'), `{"error":"${target.apiKey}"}\n`);
    assert.ok(holders(target.apiKey).includes(path.join('data', 'requests-planted.jsonl')));
});
