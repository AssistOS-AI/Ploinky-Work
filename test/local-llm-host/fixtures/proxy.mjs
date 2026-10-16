// Shared by the gate, route and proxy tests: a local-llm host with the stand-in runner (no network, no GPU, no installer) in its own
// temporary home, and a proxy server around it. Nothing here touches the real home: every home is a scratch folder.
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';

import { createLocalLlmHost } from '../../../lib/local-llm-host/host.mjs';
import { createPworkerServer } from '../../../lib/server.mjs';
import { MODEL, SNAPSHOT, fakeAdapter, freePort } from './engine.mjs';

/** A temporary home, and `later(fn)`: what must run before the folder is removed (the hooks of one test run first-registered first, so the folder cannot be removed by its own hook). */
function ownedHome(t) {
    const home = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), 'lh-'));
    const cleanups = [];
    t.after(async () => {
        for (const cleanup of cleanups.reverse()) { try { await cleanup(); } catch { /* the folder goes anyway */ } }
        fs.rmSync(home, { recursive: true, force: true });
    });
    return { home, later: (fn) => cleanups.push(fn) };
}

/** A stand-in runner adapter that counts its starts. */
export function countingAdapter() {
    const adapter = fakeAdapter();
    const counter = { starts: 0, downloads: 0 };
    return { counter, adapter: Object.freeze({ ...adapter, async start(ctx) { counter.starts += 1; return adapter.start(ctx); } }) };
}

/**
 * The seams that make a host run as on Linux on any machine, with the stand-in runner. `weights`: the state of the model's weights
 * on disk ('complete' or 'absent'); `installed`: whether the runner counts as installed.
 */
export async function hostSeams(home, { weights = 'complete', installed = true, counter = { starts: 0, downloads: 0 }, adapter = fakeAdapter(), extra = {} } = {}) {
    fs.mkdirSync(path.join(home, 'proc'), { recursive: true });
    return {
        platformName: 'linux', procDir: path.join(home, 'proc'), ownerPort: await freePort(),
        controllerOptions: {
            runners: { 'fake-gguf': adapter }, snapshot: async () => structuredClone(SNAPSHOT),
            download: async ({ artifact }) => { counter.downloads += 1; return { status: 'complete', path: `/models/${artifact.file}`, bytesTransferred: 0 }; },
            inspect: async () => (weights === 'complete' ? { state: 'complete', bytes: 1 } : { state: 'absent', bytes: 0 }),
            remove: async () => 1,
            detectRunner: () => (installed ? { installed: true, version: 'v1', reason: null } : { installed: false, version: null, reason: 'Not installed.' }),
            pollMs: 5, stopGraceMs: 500, profile: 'dedicated', ...extra,
        },
    };
}

/** A host in its own temporary home that owns it and has the stand-in runner; drained when the test ends. */
export async function startHost(t, { config = {}, env = {}, ...options } = {}) {
    const { home, later } = ownedHome(t);
    const counting = countingAdapter();
    const host = createLocalLlmHost({ config, home, env: { PWORKER_HOME: home, ...env }, ...(await hostSeams(home, { counter: counting.counter, adapter: counting.adapter, ...options })) });
    later(() => host.drain());
    await host.acquire();
    return { host, home, counter: counting.counter };
}

/** A fallback provider: answers every chat completion with `fallback answer`, and lists no models. */
export const fallbackFetch = (calls = []) => async (url, init = {}) => {
    calls.push({ url: String(url), method: init.method ?? 'GET' });
    if (String(url).endsWith('/v1/models')) return Response.json({ data: [] });
    return Response.json({ choices: [{ message: { content: 'fallback answer' }, finish_reason: 'stop' }], usage: { prompt_tokens: 1, completion_tokens: 2 } });
};

/**
 * A proxy server with a local-llm provider (and a fallback provider `fb`), around a host in its own home, listening on a free port.
 * Tiers: `tiny` = engine then fallback, `solo` = engine alone; both name the stand-in runner so the gate does not choose by profile.
 */
export async function startProxy(t, { localLlm = {}, hostOptions = {}, config = {}, tiers = null, coreOptions = {}, env = {} } = {}) {
    const { home, later } = ownedHome(t);
    const counting = countingAdapter();
    const seams = await hostSeams(home, { counter: counting.counter, adapter: counting.adapter, ...hostOptions });
    const entry = { upstream: 'local-llm', model: MODEL, runner: 'fake-gguf' };
    const fullConfig = {
        defaultUpstream: 'fb', dataDir: path.join(home, 'data'), taskHome: home, localLlm,
        providers: {
            'local-llm': { kind: 'local-llm', baseUrl: 'http://localhost', noKey: true, formats: { openai: '/v1/chat/completions' }, limits: { maxConcurrent: 32 }, retry: { max: 0, max5xx: 0 } },
            fb: { baseUrl: 'http://fb.test', noKey: true, formats: { openai: '/v1/chat/completions' }, limits: { maxConcurrent: 32 }, retry: { max: 0, max5xx: 0 } },
        },
        tiers: tiers ?? { tiny: [entry, { upstream: 'fb', model: 'fb-model' }], solo: [entry] },
        policy: { allowedPurposes: ['test:*'], untaggedDailyMax: 1000 },
        ...config,
    };
    const calls = [];
    const server = await createPworkerServer({
        config: fullConfig, env: { PWORKER_HOME: home, ...env },
        coreOptions: { fetchImpl: fallbackFetch(calls), ...coreOptions }, localLlmOptions: seams,
    });
    const address = await server.listen(0);
    later(() => server.close());
    const base = `http://127.0.0.1:${address.port}`;
    return { server, home, base, calls, counter: counting.counter, host: server.localLlm, later };
}

/** A JSON request to a base URL over a real connection; resolves { status, headers, body (parsed or text), text }. */
export function request(base, route, { method = 'GET', body = undefined, headers = {} } = {}) {
    return new Promise((resolve, reject) => {
        const payload = body === undefined ? null : typeof body === 'string' ? body : JSON.stringify(body);
        const req = http.request(`${base}${route}`, { method, headers: { ...(payload !== null ? { 'content-type': 'application/json', 'content-length': Buffer.byteLength(payload) } : {}), 'x-pworker-purpose': 'test:local-llm', ...headers } }, (res) => {
            const chunks = [];
            res.on('data', (chunk) => chunks.push(chunk));
            res.on('end', () => {
                const text = Buffer.concat(chunks).toString('utf8');
                let parsed = text;
                try { parsed = JSON.parse(text); } catch { /* not JSON: the text */ }
                resolve({ status: res.statusCode, headers: res.headers, body: parsed, text });
            });
        });
        req.on('error', reject);
        if (payload !== null) req.write(payload);
        req.end();
    });
}

export const chat = (base, model, extra = {}, headers = {}) => request(base, '/v1/chat/completions', { method: 'POST', body: { model, messages: [{ role: 'user', content: 'hi' }], ...extra }, headers });

export { MODEL, ownedHome };
