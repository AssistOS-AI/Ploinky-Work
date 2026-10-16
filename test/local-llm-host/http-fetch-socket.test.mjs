// httpFetch(url, { socketPath }): the request goes to a Unix socket, including a streamed answer, and the shared connection pool
// never mixes a socket with a TCP host or with another socket.
import '../home.mjs';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { httpFetch } from '../../lib/http-fetch.mjs';

function scratch(t) {
    const dir = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), 'hf-'));
    t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
    return dir;
}

function listenOn(t, file, handler) {
    const server = http.createServer(handler);
    t.after(() => { server.closeAllConnections(); server.close(); });
    return new Promise((resolve, reject) => { server.once('error', reject); server.listen(file, () => resolve(server)); });
}

test('a GET and a POST reach the server on the socket, with the URL path and the request headers', async (t) => {
    const dir = scratch(t);
    const file = path.join(dir, 'a.sock');
    const seen = [];
    await listenOn(t, file, (req, res) => {
        let body = '';
        req.on('data', (chunk) => { body += chunk; });
        req.on('end', () => {
            seen.push({ method: req.method, url: req.url, host: req.headers.host, auth: req.headers.authorization, body });
            res.setHeader('content-type', 'application/json');
            res.end(JSON.stringify({ ok: true }));
        });
    });
    const get = await httpFetch('http://localhost/health', { socketPath: file });
    assert.equal(get.status, 200);
    assert.deepEqual(await get.json(), { ok: true });
    const post = await httpFetch('http://localhost/v1/chat/completions?x=1', {
        method: 'POST', socketPath: file, headers: { authorization: 'Bearer k', 'content-type': 'application/json' }, body: '{"model":"m"}',
    });
    assert.equal(post.status, 200);
    await post.text();
    assert.deepEqual(seen.map(({ method, url, auth, body }) => ({ method, url, auth, body })), [
        { method: 'GET', url: '/health', auth: undefined, body: '' },
        { method: 'POST', url: '/v1/chat/completions?x=1', auth: 'Bearer k', body: '{"model":"m"}' },
    ]);
    assert.equal(seen[0].host, 'localhost');
});

test('a streamed answer arrives chunk by chunk, before the server has finished', async (t) => {
    const dir = scratch(t);
    const file = path.join(dir, 's.sock');
    let release;
    const gate = new Promise((resolve) => { release = resolve; });
    await listenOn(t, file, (req, res) => {
        res.writeHead(200, { 'content-type': 'text/event-stream' });
        res.write('data: one\n\n');
        gate.then(() => { res.write('data: two\n\n'); res.end('data: [DONE]\n\n'); });
    });
    const response = await httpFetch('http://localhost/v1/chat/completions', { method: 'POST', body: '{}', socketPath: file });
    assert.equal(response.headers.get('content-type'), 'text/event-stream');
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    const first = await reader.read();
    assert.equal(decoder.decode(first.value), 'data: one\n\n', 'the first event arrives while the server still holds the rest');
    release();
    let rest = '';
    for (let part = await reader.read(); !part.done; part = await reader.read()) rest += decoder.decode(part.value);
    assert.equal(rest, 'data: two\n\ndata: [DONE]\n\n');
});

test('two sockets and a TCP server under the same URL host never share a connection', async (t) => {
    const dir = scratch(t);
    const answer = (name) => (req, res) => res.end(name);
    const one = path.join(dir, '1.sock');
    const two = path.join(dir, '2.sock');
    await listenOn(t, one, answer('one'));
    await listenOn(t, two, answer('two'));
    const tcp = await new Promise((resolve) => { const s = http.createServer(answer('tcp')); t.after(() => { s.closeAllConnections(); s.close(); }); s.listen(0, '127.0.0.1', () => resolve(s)); });
    const read = async (url, init) => (await httpFetch(url, init)).text();
    // Repeated, so a pooled keep-alive connection would be reused if the pool keyed them alike.
    for (let round = 0; round < 3; round += 1) {
        assert.equal(await read('http://localhost/', { socketPath: one }), 'one');
        assert.equal(await read('http://localhost/', { socketPath: two }), 'two');
        assert.equal(await read(`http://127.0.0.1:${tcp.address().port}/`), 'tcp');
    }
});

test('a missing socket rejects as a failed fetch, and an abort ends a request that waits', async (t) => {
    const dir = scratch(t);
    await assert.rejects(httpFetch('http://localhost/', { socketPath: path.join(dir, 'none.sock') }), (error) => error instanceof TypeError && error.cause?.code === 'ENOENT');
    const file = path.join(dir, 'slow.sock');
    await listenOn(t, file, () => {});
    const controller = new AbortController();
    const pending = httpFetch('http://localhost/', { socketPath: file, signal: controller.signal });
    setTimeout(() => controller.abort(), 20);
    await assert.rejects(pending, (error) => error.name === 'AbortError');
});
