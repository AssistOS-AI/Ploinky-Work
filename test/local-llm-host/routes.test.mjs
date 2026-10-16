// The admin routes of the engine, /v1/local-llm/*, on a real proxy server: the guard, the JSON contract, the duplicate request id,
// the unknown model, the overview with the runners this machine cannot use, and the unavailable engine as data.
import '../home.mjs';
import assert from 'node:assert/strict';
import { Readable } from 'node:stream';
import test from 'node:test';

import { LocalLlmError } from '../../lib/local-llm/errors.mjs';
import { createLocalLlmRoutes, statusFor } from '../../lib/local-llm-host/routes.mjs';
import { MODEL, until } from './fixtures/engine.mjs';
import { request, startProxy } from './fixtures/proxy.mjs';

const post = (proxy, route, body, headers) => request(proxy.base, `/v1/local-llm/${route}`, { method: 'POST', body, headers });

test('a foreign Host header is refused with 403 before any engine route runs, and no engine is created', async (t) => {
    const proxy = await startProxy(t);
    for (const [method, route] of [['GET', ''], ['GET', '/status'], ['POST', '/run']]) {
        const answer = await request(proxy.base, `/v1/local-llm${route}`, { method, body: method === 'POST' ? { modelId: MODEL } : undefined, headers: { host: 'evil.example' } });
        assert.equal(answer.status, 403, route);
        assert.equal(answer.body.error.type, 'forbidden');
    }
    assert.equal(proxy.host.current(), null);
});

test('the proxy token guards the routes like every other endpoint', async (t) => {
    const proxy = await startProxy(t, { env: { PWORKER_TOKEN: 'proxy-token-for-the-test' } });
    assert.equal((await request(proxy.base, '/v1/local-llm')).status, 401);
    assert.equal((await request(proxy.base, '/v1/local-llm', { headers: { authorization: 'Bearer proxy-token-for-the-test' } })).status, 200);
});

test('a body that is not JSON, or not an object, is a 400 and nothing runs', async (t) => {
    const proxy = await startProxy(t);
    const bad = await post(proxy, 'run', '{"modelId": ');
    assert.equal(bad.status, 400);
    assert.equal(bad.body.error.type, 'invalid_json');
    const array = await post(proxy, 'run', '[1,2]');
    assert.equal(array.status, 400);
    assert.equal(proxy.host.current(), null, 'no engine was created for a malformed request');
    assert.equal((await post(proxy, 'no-such-route', {})).status, 404);
    assert.equal((await request(proxy.base, '/v1/local-llm/run')).status, 404, 'a GET of an action is not found');
});

test('a body above 32 MiB is a 413', async () => {
    const routes = createLocalLlmRoutes({ host: { redact: (text) => text, current: () => null, controller: async () => { throw new Error('must not be reached'); }, describe: () => ({ runners: [], ownership: {} }) } });
    const sent = [];
    const req = Object.assign(Readable.from((function* () { for (let i = 0; i < 40; i++) yield Buffer.alloc(1024 * 1024, 97); })()), { method: 'POST' });
    await routes(req, {}, new URL('http://x/v1/local-llm/run'), { sendJson: (res, status, body) => sent.push([status, body]) });
    assert.equal(sent[0][0], 413);
    assert.equal(sent[0][1].error.type, 'request_too_large');
});

test('run: accepted once, a repeated request id is answered duplicate:true, and an unknown model is a 404', async (t) => {
    const proxy = await startProxy(t);
    const first = await post(proxy, 'run', { modelId: MODEL, runnerId: 'fake-gguf', requestId: 'request-routes-01' });
    assert.equal(first.status, 202, first.text);
    assert.equal(first.body.accepted, true);
    const again = await post(proxy, 'run', { modelId: MODEL, runnerId: 'fake-gguf', requestId: 'request-routes-01' });
    assert.equal(again.status, 202);
    assert.equal(again.body.duplicate, true);
    assert.equal(proxy.counter.starts <= 1, true, 'the repeat did not start a second run');
    await until(async () => (await request(proxy.base, '/v1/local-llm/status')).body.phase === 'ready');
    const status = await request(proxy.base, '/v1/local-llm/status');
    assert.equal(status.body.available, true);
    assert.equal(status.body.deployment.modelId, MODEL);
    assert.equal(status.body.startedBy, 'admin');
    assert.equal(proxy.counter.starts, 1);
    const unknown = await post(proxy, 'run', { modelId: 'no-such-model', runnerId: 'fake-gguf', requestId: 'request-routes-02', replace: true });
    assert.equal(unknown.status, 404);
    assert.equal(unknown.body.error.type, 'unknown_model');
    const stopped = await post(proxy, 'stop', {});
    assert.equal(stopped.status, 200);
    assert.equal(proxy.host.current().state.deployment.phase, 'idle');
    assert.equal((await post(proxy, 'cancel', {})).status, 409, 'there is no download to cancel');
});

test('the overview lists the runners this machine cannot use with their reasons, next to the ones it can', async (t) => {
    const proxy = await startProxy(t);
    const overview = await request(proxy.base, '/v1/local-llm');
    assert.equal(overview.status, 200);
    assert.equal(overview.body.available, true);
    assert.ok(overview.body.runners.some((runner) => runner.id === 'fake-gguf' && runner.installed === true));
    const unsupported = Object.fromEntries(overview.body.unsupportedRunners.map(({ id, reason }) => [id, reason]));
    for (const id of ['vllm', 'ollama', 'llama.cpp-vulkan']) assert.match(unsupported[id], /not offered by the standalone pworker on Linux/, id);
    assert.match(unsupported['llama.cpp'], /needs an NVIDIA GPU/);
    assert.ok(overview.body.models.some((model) => model.id === MODEL));
    const preview = await post(proxy, 'preview', { modelId: MODEL, runnerId: 'fake-gguf' });
    assert.equal(preview.status, 200);
    assert.ok(preview.body.preview.admission.status);
});

test('an engine that cannot be had is data for a read and an error for an action', async (t) => {
    const proxy = await startProxy(t, { hostOptions: {}, config: { localLlm: { enabled: false } } });
    const overview = await request(proxy.base, '/v1/local-llm');
    assert.equal(overview.status, 200);
    assert.equal(overview.body.available, false);
    assert.equal(overview.body.refusal.code, 'local_llm_disabled');
    assert.ok(Array.isArray(overview.body.unsupportedRunners));
    const status = await request(proxy.base, '/v1/local-llm/status');
    assert.equal(status.body.available, false);
    const run = await post(proxy, 'run', { modelId: MODEL, requestId: 'request-routes-03' });
    assert.equal(run.status, 409);
    assert.equal(run.body.error.type, 'local_llm_disabled');
});

test('error messages are redacted: the Hugging Face token and per-start keys never leave in an error body', async () => {
    const secret = 'hf_synthetic_token_for_the_redaction_test';
    const host = {
        current: () => ({}), describe: () => ({ runners: [], ownership: {} }),
        redact: (text) => String(text).split(secret).join('***'),
        controller: async () => ({ lookupModel: async () => { throw new LocalLlmError('lookup_failed', `Hugging Face refused ${secret}`); } }),
    };
    const routes = createLocalLlmRoutes({ host });
    const sent = [];
    const req = Object.assign(Readable.from([Buffer.from('{"repo":"a/b"}')]), { method: 'POST' });
    await routes(req, {}, new URL('http://x/v1/local-llm/models/lookup'), { sendJson: (res, status, body) => sent.push([status, body]) });
    assert.equal(sent[0][0], 502);
    assert.equal(sent[0][1].error.type, 'lookup_failed');
    assert.equal(JSON.stringify(sent), JSON.stringify(sent).replace(secret, ''), 'the token is not in the answer');
    assert.match(sent[0][1].error.message, /\*\*\*/);
});

test('every engine error code has an HTTP status: 404, 409, 422, 400, 502 or 503, and 500 only for the unknown', () => {
    assert.deepEqual(['unknown_model', 'busy', 'admission_incompatible', 'invalid_params', 'lookup_failed', 'owner_held', 'something_else'].map(statusFor), [404, 409, 422, 400, 502, 503, 500]);
});

test('models add, update and remove go through the registry routes (a bad entry is a 400)', async (t) => {
    const proxy = await startProxy(t);
    const bad = await post(proxy, 'models/add', { id: 'x' });
    assert.ok([400, 422].includes(bad.status), `${bad.status} ${bad.text}`);
    assert.equal(bad.body.error.type.startsWith('invalid_') || bad.body.error.type.startsWith('admission_'), true, bad.text);
    const gone = await post(proxy, 'models/remove', { modelId: 'no-such-model' });
    assert.equal(gone.status, 404);
    const seed = await post(proxy, 'models/remove', { modelId: MODEL });
    assert.equal(seed.status, 409, 'a seed model is read-only');
});


test('a runner log line or a deployment field that echoes the key of the run or the Hugging Face token leaves the status route as ***', async () => {
    const key = 'per-start-key-0123456789abcdef', token = 'hf_synthetic_token_for_the_status_scrub';
    const controller = {
        status: async () => ({ phase: 'ready', deployment: { id: 'd1', modelId: 'm', note: `launched with --api-key ${key}` }, logs: [{ seq: 1, text: `argv: llama-server --api-key ${key} HF_TOKEN=${token}` }], nextSeq: 2 }),
        chatTarget: () => ({ apiKey: key }),
    };
    const host = { redact: (text) => text, secretValues: () => [token], current: () => controller, controller: async () => controller, describe: () => ({ runners: [], ownership: { state: 'owned' } }) };
    const routes = createLocalLlmRoutes({ host });
    const sent = [];
    await routes({ method: 'GET' }, {}, new URL('http://x/v1/local-llm/status'), { sendJson: (res, status, body) => sent.push([status, body]) });
    const text = JSON.stringify(sent);
    assert.equal(sent[0][0], 200);
    assert.ok(!text.includes(key) && !text.includes(token), 'neither secret is in the answer');
    assert.equal(sent[0][1].logs[0].text, 'argv: llama-server --api-key *** HF_TOKEN=***');
    assert.equal(sent[0][1].phase, 'ready');
});
