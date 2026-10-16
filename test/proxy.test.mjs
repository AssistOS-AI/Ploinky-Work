import './home.mjs';
import test, { before, after, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { EventEmitter } from 'node:events';
import { spawn as realSpawn } from 'node:child_process';
import { mkdtempSync, readFileSync, readdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createProxy } from '../lib/core.mjs';

// Launcher safety. The local-server tests run a starter over fake children whose PIDs are small numbers (1, 1901, 1902). A signal sent
// through the default process.kill would reach whatever owns that PID, and process.kill(-1, ...) reaches every process of the user.
// So every starter in this file gets a recording killImpl (launcherHarness), and process.kill itself is replaced for the whole file by a
// guard that throws and records; afterEach fails any test that reached it. The negative-control test builds a starter WITHOUT killImpl
// to show that the guard catches it and that nothing is signalled.
const realKill = process.kill;
const guardHits = [];
before(() => {
  process.kill = function blockedKill(pid, sig) {
    guardHits.push([pid, sig]);
    throw Object.assign(new Error(`process.kill(${pid}, ${sig}) is blocked in the launcher tests`), { code: 'EPERM' });
  };
});
after(() => { process.kill = realKill; });
afterEach(() => { assert.deepEqual(guardHits.splice(0), [], 'a launcher test reached the process.kill guard (no killImpl injected)'); });

// A real file for `bin` and `gguf` (the starter only checks that they exist; the children are fakes).
const FAKE_BIN = process.execPath, FAKE_GGUF = fileURLToPath(import.meta.url);

/**
 * Recording signals for local-server tests. `killImpl` records [pid, signal] and delivers the signal to the tracked child of that PID
 * (|pid|, because a group signal is negative); `fake(pid, opts)` makes a fake child that exits `exitDelayMs` after a signal it honors
 * (SIGTERM is ignored with `ignoreTerm`). The fake's own kill() is the fallback the launcher uses when killImpl throws; it only records,
 * so a test that expects no fallback asserts `fallbackKills` is empty. `trackReal` registers a REAL child that the test itself spawned:
 * its signals go through the ChildProcess handle (never process.kill), so the guard does not see them.
 */
function launcherHarness() {
  const signals = [], fallbackKills = [], children = new Map();
  const killImpl = (pid, sig) => { signals.push([pid, sig]); children.get(Math.abs(pid))?.deliver(sig); };
  const fake = (pid, { exitDelayMs = 1, ignoreTerm = false, onExit = null } = {}) => {
    const c = new EventEmitter();
    c.pid = pid; c.exitCode = null;
    c.kill = (sig) => { fallbackKills.push([pid, sig]); };
    c.deliver = (sig) => {
      if (c.exitCode !== null || (sig === 'SIGTERM' && ignoreTerm)) return;
      setTimeout(() => { if (c.exitCode !== null) return; c.exitCode = 0; onExit?.(); c.emit('exit', 0, null); }, exitDelayMs);
    };
    children.set(pid, c);
    return c;
  };
  const trackReal = (c) => { c.deliver = (sig) => c.kill(sig); children.set(c.pid, c); return c; };
  return { signals, fallbackKills, killImpl, fake, trackReal };
}

const KEY = 'sk-test-SECRET-KEY-123456';
const MODELS = { object: 'list', data: [{ id: 'm1', pricing: { prompt: '0.000001', completion: '0.000002', cache_read: '0.0000001' }, quota_multiplier: 2, context_length: 1000 }] };

function listen(server) {
  return new Promise((res) => server.listen(0, '127.0.0.1', () => res(server.address().port)));
}
const body = (req) => new Promise((res) => { const c = []; req.on('data', (d) => c.push(d)); req.on('end', () => res(Buffer.concat(c).toString())); });

async function setup({ stub, limits = { maxConcurrent: 2, maxPerSecond: 50 }, token = null, plan = null, compare = undefined, creditBalanceModels = [], dataDir: reuseDir = null, up: reuseUp = null } = {}) {
  const seen = [];
  const up = http.createServer(async (req, res) => {
    const b = await body(req);
    seen.push({ method: req.method, url: req.url, auth: req.headers.authorization, body: b });
    if (req.url === '/v1/models') { res.writeHead(200, { 'content-type': 'application/json' }); return res.end(JSON.stringify(MODELS)); }
    stub(req, res, b, seen);
  });
  const uport = await listen(up);
  const dataDir = reuseDir || mkdtempSync(join(tmpdir(), 'pworker-'));
  const config = { defaultUpstream: 'stub', modelsCacheSeconds: 600, compare, upstreams: { stub: { plan, creditBalanceModels, baseUrl: `http://127.0.0.1:${uport}`, keyVar: 'STUB_KEY', limits, retry: { max: 3, baseMs: 20, maxWaitMs: 2000 }, formats: { openai: '/v1/chat/completions', anthropic: '/v1/messages' } } } };
  const p = createProxy({ config, env: { STUB_KEY: KEY }, dataDir, proxyToken: token, cacheDir: dataDir + '-cache' });
  const port = await listen(p.server);
  const base = `http://127.0.0.1:${port}`;
  const close = async () => { p.server.close(); up.close(); up.closeAllConnections?.(); p.server.closeAllConnections?.(); if (!reuseDir) { rmSync(dataDir, { recursive: true, force: true }); rmSync(dataDir + '-cache', { recursive: true, force: true }); } };
  return { base, seen, dataDir, close, p };
}
const logText = (dir) => readdirSync(dir).map((f) => readFileSync(join(dir, f), 'utf8')).join('');
const json = (res, status, obj, h = {}) => { res.writeHead(status, { 'content-type': 'application/json', ...h }); res.end(JSON.stringify(obj)); };

test('openai passthrough, upstream auth, usage logging, key never leaks', async () => {
  const t = await setup({ stub: (req, res) => json(res, 200, { id: 'x', choices: [], usage: { prompt_tokens: 100, completion_tokens: 20, prompt_tokens_details: { cached_tokens: 40 } } }, { 'x-ratelimit-remaining': '9' }) });
  try {
    const r = await fetch(t.base + '/v1/chat/completions', { method: 'POST', headers: { 'content-type': 'application/json', authorization: 'Bearer client-token' }, body: JSON.stringify({ model: 'm1', messages: [{ role: 'user', content: 'hi' }] }) });
    assert.equal(r.status, 200);
    assert.equal(r.headers.get('x-ratelimit-remaining'), '9');
    assert.equal((await r.json()).usage.prompt_tokens, 100);
    const call = t.seen.find((s) => s.url === '/v1/chat/completions');
    assert.equal(call.auth, 'Bearer ' + KEY);
    const rec = JSON.parse(logText(t.dataDir).trim().split('\n')[0]);
    assert.equal(rec.model, 'm1'); assert.equal(rec.in_tokens, 100); assert.equal(rec.out_tokens, 20); assert.equal(rec.cached_tokens, 40);
    assert.equal(rec.rate_headers['x-ratelimit-remaining'], '9');
    assert.ok(!logText(t.dataDir).includes(KEY));
    const stats = await (await fetch(t.base + '/stats')).json();
    assert.equal(stats.windows.minute.calls, 1);
    assert.equal(stats.windows.minute.plan_requests, 2);
    assert.ok(Math.abs(stats.windows.minute.cost_usd - (60 * 1e-6 + 40 * 1e-7 + 20 * 2e-6)) < 1e-12);
    assert.ok(!JSON.stringify(stats).includes(KEY));
    const dash = await (await fetch(t.base + '/')).text();
    assert.ok(dash.includes('<title>') && !dash.includes(KEY));
  } finally { await t.close(); }
});

test('models list is passed through and cached; health hides the key', async () => {
  const t = await setup({ stub: (q, res) => json(res, 200, {}) });
  try {
    const a = await (await fetch(t.base + '/v1/models')).json();
    await fetch(t.base + '/v1/models');
    assert.equal(a.data[0].quota_multiplier, 2);
    assert.equal(t.seen.filter((s) => s.url === '/v1/models').length, 1);
    assert.equal(t.seen[0].auth, `Bearer ${KEY}`);
    const h = await (await fetch(t.base + '/health')).text();
    assert.ok(h.includes('"key_configured":true') && !h.includes(KEY));
  } finally { await t.close(); }
});

test('openai streaming: SSE passthrough, include_usage injected, TTFT and usage logged', async () => {
  const t = await setup({ stub: async (req, res, b) => {
    assert.equal(JSON.parse(b).stream_options.include_usage, true);
    res.writeHead(200, { 'content-type': 'text/event-stream' });
    res.write('data: {"choices":[{"delta":{"content":"he"}}]}\n\n');
    await new Promise((r) => setTimeout(r, 30));
    res.write('data: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":3}}\n\n');
    res.end('data: [DONE]\n\n');
  } });
  try {
    const r = await fetch(t.base + '/v1/chat/completions', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ model: 'm1', stream: true, messages: [] }) });
    const text = await r.text();
    assert.ok(text.includes('"content":"he"') && text.includes('[DONE]'));
    const rec = JSON.parse(logText(t.dataDir).trim().split('\n')[0]);
    assert.equal(rec.stream, true); assert.equal(rec.in_tokens, 7); assert.equal(rec.out_tokens, 3);
    assert.ok(rec.ttft_ms >= 0 && rec.ttft_ms < rec.latency_ms);
  } finally { await t.close(); }
});

test('anthropic passthrough, non-stream and stream', async () => {
  const t = await setup({ stub: (req, res, b) => {
    if (JSON.parse(b).stream) {
      res.writeHead(200, { 'content-type': 'text/event-stream' });
      res.write('event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":11,"cache_read_input_tokens":5,"output_tokens":1}}}\n\n');
      res.end('event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":9}}\n\n');
    } else json(res, 200, { type: 'message', usage: { input_tokens: 4, output_tokens: 2 } });
  } });
  try {
    const h = { 'content-type': 'application/json', 'anthropic-version': '2023-06-01', 'x-api-key': 'client' };
    const a = await (await fetch(t.base + '/v1/messages', { method: 'POST', headers: h, body: JSON.stringify({ model: 'm1', messages: [] }) })).json();
    assert.equal(a.type, 'message');
    const s = await (await fetch(t.base + '/v1/messages', { method: 'POST', headers: h, body: JSON.stringify({ model: 'm1', stream: true, messages: [] }) })).text();
    assert.ok(s.includes('message_delta'));
    const recs = logText(t.dataDir).trim().split('\n').map((l) => JSON.parse(l));
    assert.equal(recs[0].format, 'anthropic'); assert.equal(recs[0].in_tokens, 4);
    assert.equal(recs[1].in_tokens, 11); assert.equal(recs[1].out_tokens, 9); assert.equal(recs[1].cached_tokens, 5);
    assert.equal(t.seen.find((x) => x.url === '/v1/messages').auth, 'Bearer ' + KEY);
  } finally { await t.close(); }
});

test('429 is retried honouring retry-after, then succeeds; attempts are logged and inferred', async () => {
  let n = 0;
  const t = await setup({ stub: (req, res) => {
    if (++n <= 2) return json(res, 429, { error: 'slow down ' + KEY }, { 'retry-after': '0.05', 'x-ratelimit-limit-hour': '100' });
    json(res, 200, { usage: { prompt_tokens: 1, completion_tokens: 1 } });
  } });
  try {
    const t0 = Date.now();
    const r = await fetch(t.base + '/v1/chat/completions', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ model: 'm1', messages: [] }) });
    assert.equal(r.status, 200);
    assert.equal(n, 3);
    assert.ok(Date.now() - t0 >= 80);
    const recs = logText(t.dataDir).trim().split('\n').map((l) => JSON.parse(l));
    assert.deepEqual(recs.map((x) => x.status), [429, 429, 200]);
    assert.deepEqual(recs.map((x) => x.attempt), [1, 2, 3]);
    assert.ok(!logText(t.dataDir).includes(KEY));
    const s = await (await fetch(t.base + '/stats')).json();
    assert.equal(s.inferred_limits.r429_count, 2);
    assert.equal(s.inferred_limits.headers_seen['x-ratelimit-limit-hour'].count, 2);
  } finally { await t.close(); }
});

test('persistent 429 is returned to the client, not swallowed', async () => {
  const t = await setup({ stub: (req, res) => json(res, 429, { error: 'no' }, { 'retry-after': '0.01' }) });
  try {
    const r = await fetch(t.base + '/v1/chat/completions', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ model: 'm1', messages: [] }) });
    assert.equal(r.status, 429);
    assert.equal(logText(t.dataDir).trim().split('\n').length, 4);
  } finally { await t.close(); }
});

test('limiter bounds concurrency and exposes queue depth', async () => {
  let active = 0, maxActive = 0;
  const t = await setup({ limits: { maxConcurrent: 1, maxPerSecond: 100 }, stub: async (req, res) => {
    active++; maxActive = Math.max(maxActive, active);
    await new Promise((r) => setTimeout(r, 40));
    active--; json(res, 200, { usage: {} });
  } });
  try {
    const go = () => fetch(t.base + '/v1/chat/completions', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ model: 'm1', messages: [] }) });
    const all = [go(), go(), go()];
    await new Promise((r) => setTimeout(r, 15));
    const s = await (await fetch(t.base + '/stats')).json();
    assert.ok(s.upstreams.stub.depth >= 1);
    assert.deepEqual((await Promise.all(all)).map((r) => r.status), [200, 200, 200]);
    assert.equal(maxActive, 1);
  } finally { await t.close(); }
});

test('client token is enforced when configured; health stays open', async () => {
  const t = await setup({ token: 'local-tok', stub: (q, res) => json(res, 200, {}) });
  try {
    assert.equal((await fetch(t.base + '/stats')).status, 401);
    assert.equal((await fetch(t.base + '/health')).status, 200);
    assert.equal((await fetch(t.base + '/stats', { headers: { authorization: 'Bearer local-tok' } })).status, 200);
  } finally { await t.close(); }
});

test('unreachable upstream gives a 502 and a log record', async () => {
  const t = await setup({ stub: (q, res) => json(res, 200, {}) });
  try {
    t.p.upstreams.stub.baseUrl = 'http://127.0.0.1:1';
    const r = await fetch(t.base + '/v1/chat/completions', { method: 'POST', headers: { 'content-type': 'application/json' }, body: '{"model":"m1"}' });
    assert.equal(r.status, 502);
    assert.equal(JSON.parse(logText(t.dataDir).trim().split('\n')[0]).status, 502);
  } finally { await t.close(); }
});

test('quota headers, per-minute limit, credit-billed model marking and quota window inference', async () => {
  let n = 0;
  const t = await setup({ limits: { maxConcurrent: 1, maxPerMinute: 15 }, stub: (req, res, b) => {
    if (JSON.parse(b).model === 'paid') return json(res, 402, { error: 'This model is billed from your credit balance. Maintain a balance of at least $1.00' });
    const rem = [10, 9.9, 9.8, 10][n++ % 4];
    json(res, 200, { usage: { prompt_tokens: 1, completion_tokens: 1 } }, { 'x-quota-remaining': String(rem), 'x-quota-cost': '0.1', 'x-ratelimit-limit': '15' });
  } });
  try {
    assert.equal(t.p.limiters.stub.maxPerMinute, 15);
    const go = (model) => fetch(t.base + '/v1/chat/completions', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ model, messages: [] }) });
    for (let i = 0; i < 4; i++) await go('m1');
    assert.equal((await go('paid')).status, 402);
    const s = await (await fetch(t.base + '/stats')).json();
    assert.equal(s.by_model.paid.billing, 'credit');
    assert.equal(s.by_model.m1.billing, 'plan');
    assert.deepEqual(s.by_model.m1.quota_cost_seen, [0.1]);
    assert.ok(Math.abs(s.by_model.m1.plan_requests - 0.4) < 1e-9);
    assert.equal(s.quota.remaining_now, 10);
    assert.equal(s.quota.resets.length, 1);
    const models = await (await fetch(t.base + '/v1/models')).json();
    assert.equal(models.data[0].x_billing, 'plan_or_unknown');
  } finally { await t.close(); }
});

test('known credit-balance models are labeled and refused before an upstream request', async () => {
  const t = await setup({ creditBalanceModels: ['paid'], stub: (req, res) => json(res, 200, { usage: {} }) });
  try {
    const models = await (await fetch(t.base + '/v1/models')).json();
    assert.equal(models.data[0].x_billing, 'plan_or_unknown');
    const r = await fetch(t.base + '/v1/chat/completions', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ model: 'paid', messages: [] }) });
    assert.equal(r.status, 402);
    assert.equal((await r.json()).error.type, 'credit_balance_required');
    assert.equal(t.seen.filter((call) => call.url === '/v1/chat/completions').length, 0);
  } finally { await t.close(); }
});

test('limiter enforces max starts per minute by waiting, never dropping', async () => {
  const { Limiter } = await import('../lib/limiter.mjs');
  const l = new Limiter({ maxConcurrent: 5, maxPerMinute: 2 });
  await l.schedule(() => 1); await l.schedule(() => 2);
  let third = false; l.schedule(() => { third = true; });
  await new Promise((r) => setTimeout(r, 30));
  assert.equal(third, false); assert.equal(l.depth, 1);
});

const post = (base, model = 'm1', extra = {}) => fetch(base + '/v1/chat/completions', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ model, messages: [], ...extra }) });
const okStub = (usage = { prompt_tokens: 1000, completion_tokens: 500, prompt_tokens_details: { cached_tokens: 200 } }) => {
  let rem = 100;
  return (req, res) => { rem -= 2; json(res, 200, { usage }, { 'x-quota-remaining': String(rem), 'x-quota-cost': '2' }); };
};

test('plan limits: calls, credits and tokens in rolling windows, provider quota next to ours, 80% warning', async () => {
  const plan = { priceUsdPerMonth: 15, limits: [
    { name: 'per_minute', unit: 'calls', window: '60s', max: 15 },
    { name: 'per_5h', unit: 'credits', window: '5h', max: 5, provider: { remainingHeader: 'x-quota-remaining', costHeader: 'x-quota-cost' } },
    { name: 'per_week', unit: 'tokens', window: '7d', max: 100000 },
  ] };
  const t = await setup({ plan, stub: okStub() });
  try {
    await post(t.base); await post(t.base);
    const s = await (await fetch(t.base + '/stats')).json();
    const [m, c, w] = s.plan.stub.limits;
    assert.equal(s.plan.stub.price_usd_per_month, 15);
    assert.equal(m.used, 2); assert.equal(m.remaining, 13);
    assert.equal(c.used, 4); assert.equal(c.remaining, 1); assert.equal(c.warn, true); assert.equal(c.exceeded, false);
    assert.equal(w.used, 3000); assert.equal(w.warn, false);
    assert.ok(c.next_relief_ms > 4 * 3600_000);
    assert.equal(c.provider.remaining, 96);
    assert.equal(c.provider.agreement.provider_used, 4); assert.equal(c.provider.agreement.our_used, 4); assert.equal(c.provider.agreement.diff, 0);
    assert.equal(s.plan.stub.warnings.length, 1);
    assert.ok(s.plan.stub.warnings[0].startsWith('per_5h'));
  } finally { await t.close(); }
});

test('queue pauses before exceeding a credits limit and never drops', async () => {
  const plan = { priceUsdPerMonth: 15, limits: [{ name: 'tiny', unit: 'credits', window: '400ms', max: 4 }] };
  const starts = [];
  const t = await setup({ plan, limits: { maxConcurrent: 1, maxPerSecond: 100 }, stub: (req, res) => { starts.push(Date.now()); json(res, 200, { usage: {} }, { 'x-quota-cost': '2' }); } });
  try {
    await fetch(t.base + '/v1/models'); // warm the model list (m1 costs 2)
    const t0 = Date.now();
    const rs = await Promise.all([post(t.base), post(t.base), post(t.base)]);
    assert.deepEqual(rs.map((r) => r.status), [200, 200, 200]);
    assert.ok(starts[2] - starts[0] >= 380, 'third call waits for the window: ' + (starts[2] - starts[0]));
    assert.ok(starts[1] - starts[0] < 300);
    assert.ok(Date.now() - t0 < 5000);
  } finally { await t.close(); }
});

test('value comparison against DeepSeek and list prices, with a verdict', async () => {
  const compare = { cheap: { inputUsdPerM: 1, outputUsdPerM: 2, cachedInputUsdPerM: 0.1 }, _note: 'ignored' };
  const plan = { priceUsdPerMonth: 15, limits: [] };
  const t = await setup({ plan, compare, stub: okStub() });
  try {
    await fetch(t.base + '/v1/models');
    await post(t.base); await post(t.base);
    const v = (await (await fetch(t.base + '/stats')).json()).value.stub;
    // per call: 800 fresh + 200 cached input, 500 output
    assert.equal(v.periods.day.tokens.in, 1600); assert.equal(v.periods.day.tokens.cached, 400); assert.equal(v.periods.day.tokens.out, 1000);
    assert.ok(Math.abs(v.periods.day.compare_usd.cheap - (1600 * 1 + 400 * 0.1 + 1000 * 2) / 1e6) < 1e-9);
    assert.ok(Math.abs(v.periods.day.openference_list_usd - (1600 * 1e-6 + 400 * 1e-7 + 1000 * 2e-6)) < 1e-9);
    assert.ok(Math.abs(v.periods.month.subscription_prorated_usd - 15) < 1e-9);
    assert.ok(Math.abs(v.periods.day.subscription_prorated_usd - 0.5) < 1e-9);
    assert.equal(v.projection.span_days, 1);
    assert.ok(v.verdict.includes('subscription costs') && v.verdict.includes('more'));
    // a heavy month flips the verdict
    const heavy = { priceUsdPerMonth: 0.0001, limits: [] };
    t.p.upstreams.stub.plan = heavy;
    const v2 = (await (await fetch(t.base + '/stats')).json()).value.stub;
    assert.ok(v2.verdict.includes('subscription saves'));
  } finally { await t.close(); }
});

test('rolling windows, queue starts and value are rebuilt from the logs after a restart', async () => {
  const dataDir = mkdtempSync(join(tmpdir(), 'pworker-restart-'));
  const plan = { priceUsdPerMonth: 15, limits: [{ name: 'per_5h', unit: 'credits', window: '5h', max: 100 }] };
  try {
    const a = await setup({ plan, dataDir, stub: okStub() });
    await fetch(a.base + '/v1/models');
    await post(a.base); await post(a.base);
    await a.close();
    const b = await setup({ plan, dataDir, stub: okStub() });
    try {
      const s = await (await fetch(b.base + '/stats')).json();
      assert.equal(s.plan.stub.limits[0].used, 4);
      assert.equal(s.value.stub.periods.day.tokens.calls, 2);
      assert.equal(b.p.limiters.stub.starts.length, 2);
    } finally { await b.close(); }
  } finally { rmSync(dataDir, { recursive: true, force: true }); }
});

// ---- fallback to a second upstream ----
async function setupFallback({ primary, fallbackStub = (req, res, b) => json(res, 200, { model: JSON.parse(b).model, usage: { prompt_tokens: 3, completion_tokens: 2, cost: 0.0001 } }), plan = null, limits = { maxConcurrent: 2, maxPerSecond: 50 }, maxWaitMs = 500, retry = { max: 1, baseMs: 10, maxWaitMs: 2000, max5xx: 1 } } = {}) {
  const seenFb = [];
  const mk = (stub, seen) => http.createServer(async (req, res) => {
    const b = await body(req);
    if (req.url === '/v1/models') return json(res, 200, MODELS);
    seen?.push({ url: req.url, auth: req.headers.authorization, body: b });
    stub(req, res, b);
  });
  const a = mk(primary), f = mk(fallbackStub, seenFb);
  const [pa, pf] = [await listen(a), await listen(f)];
  const dataDir = mkdtempSync(join(tmpdir(), 'pworker-fb-'));
  const config = {
    defaultUpstream: 'main',
    upstreams: {
      main: { plan, baseUrl: `http://127.0.0.1:${pa}`, keyVar: 'MAIN_KEY', limits, retry, formats: { openai: '/v1/chat/completions', anthropic: '/v1/messages' } },
      alt: { baseUrl: `http://127.0.0.1:${pf}`, keyVar: 'ALT_KEY', limits: { maxConcurrent: 2 }, retry: { max: 0, baseMs: 10, maxWaitMs: 100 }, formats: { openai: '/v1/chat/completions' } },
    },
    fallback: { upstream: 'alt', models: { m1: 'alt-model' }, maxWaitMs },
  };
  const p = createProxy({ config, env: { MAIN_KEY: KEY, ALT_KEY: 'alt-key-987654' }, dataDir });
  const port = await listen(p.server);
  const base = `http://127.0.0.1:${port}`;
  const close = async () => { for (const s of [p.server, a, f]) { s.close(); s.closeAllConnections?.(); } rmSync(dataDir, { recursive: true, force: true }); };
  return { base, seenFb, dataDir, close, p };
}
const recsOf = (dir) => logText(dir).trim().split('\n').map((l) => JSON.parse(l));

test('fallback: 5xx after retries goes to the fallback upstream, marked in header, log and stats', async () => {
  let n = 0;
  const t = await setupFallback({ primary: (req, res) => { n++; json(res, 503, { error: 'down' }); } });
  try {
    const r = await post(t.base);
    assert.equal(r.status, 200);
    assert.equal(r.headers.get('x-pworker-fallback'), 'alt/alt-model');
    assert.equal(r.headers.get('x-pworker-fallback-reason'), '5xx');
    assert.equal((await r.json()).model, 'alt-model');
    assert.equal(n, 2, 'one try plus max5xx=1 retry');
    assert.equal(t.seenFb[0].auth, 'Bearer alt-key-987654');
    const recs = recsOf(t.dataDir);
    assert.deepEqual(recs.map((x) => [x.upstream, x.status]), [['main', 503], ['main', 503], ['alt', 200]]);
    assert.equal(recs[1].fallback_to, 'alt/alt-model');
    assert.equal(recs[2].fallback_from, 'main'); assert.equal(recs[2].fallback_model, 'm1'); assert.equal(recs[2].fallback_kind, '5xx');
    assert.equal(recs[2].usd, 0.0001);
    const s = await (await fetch(t.base + '/stats')).json();
    assert.equal(s.fallback.total, 1); assert.deepEqual(s.fallback.by_kind, { '5xx': 1 });
    assert.equal(s.windows.minute.fallbacks, 1);
    assert.equal((await (await fetch(t.base + '/health')).json()).fallback.upstream, 'alt');
  } finally { await t.close(); }
});

test('fallback: unreachable primary, and 429 with a retry-after beyond maxWaitMs, fall back at once', async () => {
  const t = await setupFallback({ primary: (req, res) => json(res, 429, { error: 'quota' }, { 'retry-after': '3600' }) });
  try {
    const t0 = Date.now();
    const r = await post(t.base);
    assert.equal(r.status, 200); assert.equal(r.headers.get('x-pworker-fallback-reason'), '429');
    assert.ok(Date.now() - t0 < 1500);
    t.p.upstreams.main.baseUrl = 'http://127.0.0.1:1';
    t.p.limiters.main.pausedUntil = 0;
    const u = await post(t.base);
    assert.equal(u.status, 200); assert.equal(u.headers.get('x-pworker-fallback-reason'), 'unreachable');
  } finally { await t.close(); }
});

test('fallback: a plan-gate wait above maxWaitMs falls back before calling the primary', async () => {
  const plan = { limits: [{ name: 'tiny', unit: 'calls', window: '1h', max: 1 }] };
  let n = 0;
  const t = await setupFallback({ plan, primary: (req, res) => { n++; json(res, 200, { usage: {} }); } });
  try {
    assert.equal((await post(t.base)).headers.get('x-pworker-fallback'), null);
    const r = await post(t.base);
    assert.equal(r.status, 200); assert.equal(r.headers.get('x-pworker-fallback-reason'), 'wait');
    assert.equal(n, 1);
    assert.ok(recsOf(t.dataDir)[1].fallback_reason.includes('tiny'));
  } finally { await t.close(); }
});

test('fallback: opt-out header, unmapped models and the anthropic format stay on the primary', async () => {
  const t = await setupFallback({ primary: (req, res) => json(res, 503, { error: 'down' }) });
  try {
    const opt = await fetch(t.base + '/v1/chat/completions', { method: 'POST', headers: { 'content-type': 'application/json', 'x-pworker-no-fallback': '1' }, body: JSON.stringify({ model: 'm1', messages: [] }) });
    assert.equal(opt.status, 503); assert.equal(opt.headers.get('x-pworker-fallback'), null);
    assert.equal((await post(t.base, 'other')).status, 503);
    const an = await fetch(t.base + '/v1/messages', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ model: 'm1', messages: [] }) });
    assert.equal(an.status, 503);
    assert.equal(t.seenFb.length, 0);
    assert.equal(recsOf(t.dataDir).filter((r) => r.fallback_from).length, 0);
  } finally { await t.close(); }
});

test('fallback: a failing fallback upstream returns its own error, never loops', async () => {
  const t = await setupFallback({ primary: (req, res) => json(res, 500, { error: 'a' }), fallbackStub: (req, res) => json(res, 502, { error: 'b' }) });
  try {
    const r = await post(t.base);
    assert.equal(r.status, 502); assert.equal(r.headers.get('x-pworker-fallback'), 'alt/alt-model');
    assert.equal(t.seenFb.length, 1);
  } finally { await t.close(); }
});

async function tierSetup({ primary, secondary, localStart = null, audit = null }) {
  const mk = async (fn) => { const s = http.createServer(async (req, res) => { const b = await body(req); if (req.url === '/v1/models') return json(res, 200, MODELS); fn(req, res, b); }); return { s, port: await listen(s) }; };
  const a = await mk(primary), b = await mk(secondary);
  const dataDir = mkdtempSync(join(tmpdir(), 'pworker-'));
  const upA = { baseUrl: `http://127.0.0.1:${a.port}`, keyVar: 'K', limits: { maxConcurrent: 2, maxPerSecond: 50 }, retry: { max: 0, max5xx: 0, baseMs: 10, maxWaitMs: 100 }, formats: { openai: '/v1/chat/completions' } };
  const config = { defaultUpstream: 'a', modelsCacheSeconds: 600,
    upstreams: { a: upA, b: { ...upA, baseUrl: `http://127.0.0.1:${b.port}` }, ...(localStart ? { loc: { ...upA, baseUrl: 'http://127.0.0.1:1', noKey: true, keyVar: undefined, start: localStart } } : {}) },
    tiers: { tiny: localStart ? [{ upstream: 'loc', model: 'L' }, { upstream: 'b', model: 'B' }] : [{ upstream: 'b', model: 'B' }], small: [{ upstream: 'a', model: 'A' }, { upstream: 'b', model: 'B' }], good: [{ upstream: 'b', model: 'G' }] },
    audit };
  const h = launcherHarness();
  const noSpawn = () => { throw new Error('this test must not start a local server'); };
  const p = createProxy({ config, env: { K: KEY }, dataDir, auditDir: join(dataDir, 'aud'), cacheDir: dataDir + '-cache', starterOptions: { gpuPids: () => [4242], killImpl: h.killImpl, spawn: noSpawn } });
  const port = await listen(p.server);
  const close = () => { p.server.close(); a.s.close(); b.s.close(); p.server.closeAllConnections?.(); a.s.closeAllConnections?.(); b.s.closeAllConnections?.(); rmSync(dataDir, { recursive: true, force: true }); rmSync(dataDir + '-cache', { recursive: true, force: true }); };
  return { base: `http://127.0.0.1:${port}`, dataDir, close };
}
const ask = (base, model, h = {}) => fetch(base + '/v1/chat/completions', { method: 'POST', headers: { 'content-type': 'application/json', ...h }, body: JSON.stringify({ model, messages: [{ role: 'user', content: 'q' }] }) });
const echoModel = (req, res, b) => json(res, 200, { choices: [{ message: { content: 'from ' + JSON.parse(b).model } }], usage: { prompt_tokens: 1, completion_tokens: 1 } });

test('tiers: a tier resolves to its first upstream, falls back down its chain, unknown tier is an error', async () => {
  const t = await tierSetup({ primary: (req, res) => json(res, 503, { error: 'down' }), secondary: echoModel });
  try {
    let r = await ask(t.base, 'small');
    assert.equal(r.status, 200);
    assert.equal((await r.json()).choices[0].message.content, 'from B');
    assert.equal(r.headers.get('x-pworker-tier'), 'small');
    assert.equal(r.headers.get('x-pworker-model'), 'b/B');
    r = await ask(t.base, 'good');
    assert.equal((await r.json()).choices[0].message.content, 'from G');
    r = await ask(t.base, 'best');
    assert.equal(r.status, 400);
    assert.match((await r.json()).error.message, /tier "best" is not configured/);
    const models = await (await fetch(t.base + '/v1/models')).json();
    assert.ok(models.data.some((m) => m.id === 'small' && m.x_tier.serves === 'a/A'));
    const recs = recsOf(t.dataDir);
    assert.ok(recs.some((x) => x.tier === 'small' && x.upstream === 'b' && x.fallback_from === 'a'));
    const st = await (await fetch(t.base + '/stats')).json();
    assert.equal(st.last24h.by_tier.small.fallbacks, 1);
  } finally { t.close(); }
});

test('tiers: a local upstream that may not start (GPU busy) falls back without calling it', async () => {
  const t = await tierSetup({ primary: echoModel, secondary: echoModel, localStart: { bin: FAKE_BIN, gguf: FAKE_GGUF, requireFreeGpu: true } });
  try {
    const r = await ask(t.base, 'tiny');
    assert.equal(r.status, 200);
    assert.equal((await r.json()).choices[0].message.content, 'from B');
    assert.equal(r.headers.get('x-pworker-fallback-reason'), 'unavailable');
    const st = await (await fetch(t.base + '/stats')).json();
    assert.match(st.local.loc.last_refusal.reason, /GPU busy/);
  } finally { t.close(); }
});

test('audit: request/response pairs of audited tiers are stored with purpose and run tags; others are not', async () => {
  const t = await tierSetup({ primary: echoModel, secondary: echoModel, audit: { enabled: true, tiers: ['small'] } });
  try {
    await (await ask(t.base, 'small', { 'x-pworker-purpose': 'job:test', 'x-pworker-run': 'r1' })).json();
    await (await ask(t.base, 'good')).json();
    const files = readdirSync(join(t.dataDir, 'aud'));
    assert.equal(files.length, 1);
    const lines = readFileSync(join(t.dataDir, 'aud', files[0]), 'utf8').trim().split('\n').map((l) => JSON.parse(l));
    assert.equal(lines.length, 1);
    assert.equal(lines[0].purpose, 'job:test'); assert.equal(lines[0].run, 'r1'); assert.equal(lines[0].tier, 'small');
    assert.equal(lines[0].request.messages[0].content, 'q'); assert.equal(lines[0].response, 'from A');
    const st = await (await fetch(t.base + '/stats')).json();
    assert.equal(st.audit.recorded, 1); assert.equal(st.last24h.by_purpose['job:test'].calls, 1);
  } finally { t.close(); }
});

test('jobs: registered runs are refused at their budget and after finishing; untagged requests have a daily allowance; /stats per job', async () => {
  const t = await setup({ stub: (req, res) => json(res, 200, { choices: [{ message: { content: 'ok' } }], usage: { prompt_tokens: 1, completion_tokens: 1, cost: 0.01 } }) });
  t.p.guard.policy.untaggedDailyMax = 2;
  const ask = (headers = {}) => fetch(t.base + '/v1/chat/completions', { method: 'POST', headers: { 'content-type': 'application/json', ...headers }, body: JSON.stringify({ model: 'm1', messages: [{ role: 'user', content: 'hi' }] }) });
  const post = (path, b) => fetch(t.base + path, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(b) });
  try {
    assert.equal((await post('/jobs/register', { job: 'j1', run: 'r1', budget: { usd: 0.015 } })).status, 200);
    assert.equal((await post('/jobs/register', { job: 'j1', run: 'r1', budget: { usd: 1 } })).status, 400, 'a run id is registered once');
    assert.equal((await post('/jobs/register', { job: 'j1', run: 'r2' })).status, 400, 'a run needs a budget');
    const tags = { 'x-pworker-purpose': 'job:j1', 'x-pworker-run': 'r1' };
    assert.equal((await ask(tags)).status, 200);
    assert.equal((await ask(tags)).status, 200);
    const over = await ask(tags);
    assert.equal(over.status, 402);
    assert.equal((await over.json()).error.type, 'budget_exceeded');
    // untagged: two pass, the third is refused; allowed purposes are not counted
    assert.equal((await ask()).status, 200);
    assert.equal((await ask({ 'x-pworker-purpose': 'something-else' })).status, 200);
    const refused = await ask();
    assert.equal(refused.status, 403);
    assert.equal((await refused.json()).error.type, 'untagged_limit');
    assert.equal((await ask({ 'x-pworker-purpose': 'chat' })).status, 200);
    assert.equal((await post('/jobs/register', { job: 'j2', run: 'r3', budget: { calls: 10 } })).status, 200);
    assert.equal((await post('/jobs/finish', { run: 'r3', status: 'finished' })).status, 200);
    const late = await ask({ 'x-pworker-purpose': 'job:j2', 'x-pworker-run': 'r3' });
    assert.equal(late.status, 403);
    const st = await (await fetch(t.base + '/stats')).json();
    assert.equal(st.jobs.by_job.j1.calls, 2);
    assert.ok(Math.abs(st.jobs.by_job.j1.usd - 0.02) < 1e-9);
    assert.equal(st.jobs.untagged.used, 2);
    assert.equal(st.jobs.refused.budget_exceeded, 1);
    assert.equal(st.jobs.runs.find((r) => r.run === 'r3').status, 'finished');
    // a restart rebuilds the allowance from the log and keeps the registrations
    const again = createProxy({ config: { defaultUpstream: 'stub', upstreams: { stub: { baseUrl: 'http://127.0.0.1:1', limits: {} } }, jobs: { untaggedDailyMax: 2 } }, env: {}, dataDir: t.dataDir });
    assert.equal(again.guard.stats().untagged.used, 2);
    assert.equal(again.guard.runs.get('r1').budget.usd, 0.015);
  } finally { await t.close(); }
});

test('limiter: interactive priority runs before queued batch jobs, FIFO within a priority', async () => {
  const { Limiter } = await import('../lib/limiter.mjs');
  const l = new Limiter({ maxConcurrent: 1 });
  const order = [];
  let release;
  const first = l.schedule(() => new Promise((r) => { release = r; }), null, 1);
  const jobs = [['b1', 1], ['b2', 1], ['i1', 0], ['i2', 0]].map(([n, p]) => l.schedule(async () => { order.push(n); }, null, p));
  await new Promise((r) => setTimeout(r, 10)); release(); await first; await Promise.all(jobs);
  assert.deepEqual(order, ['i1', 'i2', 'b1', 'b2']);
});

test('cache: use stores and serves without a model call, strict refuses a miss, a changed prompt misses, record refreshes', async () => {
  let calls = 0;
  const t = await tierSetup({ primary: (req, res, b) => { calls += 1; echoModel(req, res, b); }, secondary: echoModel });
  try {
    const ask2 = (content, mode, model = 'small') => fetch(t.base + '/v1/chat/completions', { method: 'POST', headers: { 'content-type': 'application/json', 'x-pworker-cache': mode, 'x-pworker-purpose': 'test:cache' }, body: JSON.stringify({ model, messages: [{ role: 'user', content }], temperature: 0 }) });
    let r = await ask2('q1', 'strict');
    assert.equal(r.status, 409); assert.equal((await r.json()).error.type, 'cache_miss'); assert.equal(calls, 0);
    r = await ask2('q1', 'use'); assert.equal(r.status, 200); await r.json(); assert.equal(calls, 1);
    r = await ask2('q1', 'strict'); assert.equal(r.status, 200); assert.equal(r.headers.get('x-pworker-cache'), 'hit');
    assert.equal((await r.json()).choices[0].message.content, 'from A'); assert.equal(calls, 1);
    r = await ask2('q1 changed', 'strict'); assert.equal(r.status, 409);
    r = await ask2('q1', 'record'); await r.json(); assert.equal(calls, 2);
    r = await ask2('q1', 'use', 'good'); await r.json(); // another tier is another key
    const st = await (await fetch(t.base + '/stats')).json();
    assert.equal(st.cache.hits, 1); assert.equal(st.cache.refused, 2);
  } finally { t.close(); }
});

test('cache: a cut (finish_reason length) or empty answer is not stored', async () => {
  let calls = 0;
  const cut = (req, res, b) => { calls += 1; json(res, 200, { choices: [{ message: { content: '' }, finish_reason: 'length' }], usage: { prompt_tokens: 1, completion_tokens: 1 } }); };
  const t = await tierSetup({ primary: cut, secondary: echoModel });
  try {
    const ask3 = () => fetch(t.base + '/v1/chat/completions', { method: 'POST', headers: { 'content-type': 'application/json', 'x-pworker-cache': 'use', 'x-pworker-purpose': 'test:cache' }, body: JSON.stringify({ model: 'small', messages: [{ role: 'user', content: 'cut' }] }) });
    await (await ask3()).json(); await (await ask3()).json();
    assert.equal(calls, 2);
  } finally { t.close(); }
});

// The structure and formalizer tiers (owner 2026-10-03): JSON endpoints of a local Node service started from a script; tests use a
// fake service script, never the real models.
test('json tiers: structure and formalizer start a script upstream, forward, cache a JSON answer, refuse a chat path', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'pworker-sm-'));
  const script = join(dir, 'fake.mjs');
  const { writeFileSync } = await import('node:fs');
  writeFileSync(script, `import http from 'node:http';
const port = Number(process.argv[process.argv.indexOf('--port') + 1]);
let calls = 0;
setTimeout(() => process.exit(0), 60000); // the fixture never outlives a failed test
http.createServer(async (req, res) => {
  const c = []; for await (const x of req) c.push(x);
  const send = (o) => { res.writeHead(200, { 'content-type': 'application/json' }); res.end(JSON.stringify(o)); };
  if (req.url === '/health') return send({ ok: true });
  if (req.url === '/v1/models') return send({ object: 'list', data: [] });
  calls += 1;
  const b = JSON.parse(Buffer.concat(c).toString());
  if (req.url === '/v1/structure') return send({ object: 'structure', model: b.model, entities: { quantity: [{ text: '12' }] }, calls });
  if (req.url === '/v1/fol') return send({ object: 'fol', model: b.model, results: b.inputs.map((input) => ({ input, candidates: ['Cat(tom)'] })), calls });
}).listen(port, '127.0.0.1');
`);
  const free = await new Promise((r) => { const s = http.createServer(); s.listen(0, '127.0.0.1', () => { const p = s.address().port; s.close(() => r(p)); }); });
  const dataDir = mkdtempSync(join(tmpdir(), 'pworker-'));
  const config = { defaultUpstream: 'sm', modelsCacheSeconds: 600,
    upstreams: { sm: { baseUrl: `http://127.0.0.1:${free}`, noKey: true, limits: { maxConcurrent: 2, maxPerSecond: 50 }, retry: { max: 0, max5xx: 0, baseMs: 10, maxWaitMs: 100 }, formats: { structure: '/v1/structure', fol: '/v1/fol' },
      start: { script, requireFreeGpu: false, identity: [script], startTimeoutMs: 10000, logFile: join(dir, 'sm.log') } } },
    tiers: { structure: [{ upstream: 'sm', model: 'psm' }], formalizer: [{ upstream: 'sm', model: 'lfm' }] }, cache: { defaultMode: 'use' } };
  // A real fixture child: the starter really spawns it, and its signals are recorded and delivered through the child handle.
  const h = launcherHarness();
  const p = createProxy({ config, env: {}, dataDir, cacheDir: dataDir + '-cache', starterOptions: { killImpl: h.killImpl, spawn: (...a) => h.trackReal(realSpawn(...a)) } });
  const port = await listen(p.server);
  const post = (path, b) => fetch(`http://127.0.0.1:${port}${path}`, { method: 'POST', headers: { 'content-type': 'application/json', 'x-pworker-purpose': 'test:json-tiers' }, body: JSON.stringify(b) });
  try {
    let r = await post('/v1/structure', { model: 'structure', text: 'Ana has 12 apples.', entities: ['quantity'] });
    assert.equal(r.status, 200);
    assert.equal(r.headers.get('x-pworker-tier'), 'structure');
    assert.equal(r.headers.get('x-pworker-model'), 'sm/psm');
    let j = await r.json();
    assert.equal(j.model, 'psm'); assert.equal(j.entities.quantity[0].text, '12'); assert.equal(j.calls, 1);
    r = await post('/v1/structure', { model: 'structure', text: 'Ana has 12 apples.', entities: ['quantity'] });
    assert.equal(r.headers.get('x-pworker-cache'), 'hit');
    assert.equal((await r.json()).calls, 1);
    r = await post('/v1/fol', { model: 'formalizer', inputs: ['Tom is a cat.'] });
    j = await r.json();
    assert.equal(j.model, 'lfm'); assert.equal(j.results[0].candidates[0], 'Cat(tom)');
    r = await post('/v1/chat/completions', { model: 'structure', messages: [{ role: 'user', content: 'q' }] });
    assert.equal(r.status, 400); assert.match((await r.json()).error.message, /no endpoint \/v1\/chat\/completions/);
    const recs = recsOf(dataDir);
    assert.ok(recs.some((x) => x.tier === 'structure' && x.endpoint === '/v1/structure' && x.status === 200 && x.purpose === 'test:json-tiers'));
    assert.ok(recs.some((x) => x.upstream === 'cache' && x.endpoint === '/v1/structure'));
  } finally {
    p.server.close(); p.server.closeAllConnections?.();
    await p.starters.sm.stop();
    rmSync(dataDir, { recursive: true, force: true }); rmSync(dataDir + '-cache', { recursive: true, force: true }); rmSync(dir, { recursive: true, force: true });
  }
});

// Prompted JSON tiers (owner 2026-10-03): a tier entry with `prompt` serves /v1/structure and /v1/fol with a chat upstream and a
// template file; the reply is validated, an invalid one re-asked once, and the answer has the same contract as the dedicated models.
test('prompted tiers: structure and fol through a chat upstream, validation and one re-ask, alias tiers, cache', async () => {
  const { writeFileSync, mkdirSync } = await import('node:fs');
  const dir = mkdtempSync(join(tmpdir(), 'pworker-pr-'));
  mkdirSync(join(dir, 'prompts'));
  writeFileSync(join(dir, 'prompts', 'psm.md'), '<<<options>>>\n{"maxTokens": 500}\n<<<user>>>\nLabels:\n{{labels}}\nText: {{text}}\n<<<again>>>\nFix: {{problems}}\n');
  writeFileSync(join(dir, 'prompts', 'fol.md'), '<<<user>>>\n{{inventory}}\n{{sentences}}\n<<<again>>>\nFix: {{problems}}\n');
  let calls = 0;
  const chat = (req, res, b) => {
    calls += 1;
    const m = JSON.parse(b).messages;
    const last = m[m.length - 1].content;
    let content;
    if (last.startsWith('Fix:')) content = '{"spans": [{"label": "quantity", "text": "12 apples"}, {"label": "goal", "text": "How many"}]}';
    else if (last.includes('Labels:')) content = 'sure: {"spans": [{"label": "quantity", "text": "twelve apples"}, {"label": "person", "text": "Ana"}]}';
    else content = '{"fol": [{"s": 1, "fol": ["Has(ana, 12)"]}, {"s": 2, "fol": ["? Has(ana, 12)"]}]}';
    json(res, 200, { choices: [{ message: { content }, finish_reason: 'stop' }], usage: { prompt_tokens: 5, completion_tokens: 5 } });
  };
  const s = http.createServer(async (req, res) => { const b = await body(req); if (req.url === '/v1/models') return json(res, 200, MODELS); chat(req, res, b); });
  const port0 = await listen(s);
  const dataDir = mkdtempSync(join(tmpdir(), 'pworker-'));
  const config = { defaultUpstream: 'c', modelsCacheSeconds: 600, promptsDir: join(dir, 'prompts'), cache: { defaultMode: 'use' },
    upstreams: { c: { baseUrl: `http://127.0.0.1:${port0}`, noKey: true, limits: { maxConcurrent: 2, maxPerSecond: 50 }, retry: { max: 0, max5xx: 0, baseMs: 10, maxWaitMs: 100 }, formats: { openai: '/v1/chat/completions' } } },
    tiers: { structure: 'structure-chat', 'structure-chat': [{ upstream: 'c', model: 'Q', prompt: 'psm' }], formalizer: [{ upstream: 'c', model: 'Q', prompt: 'fol' }] } };
  const p = createProxy({ config, env: {}, dataDir, cacheDir: dataDir + '-cache' });
  const port = await listen(p.server);
  const post = (path, b) => fetch(`http://127.0.0.1:${port}${path}`, { method: 'POST', headers: { 'content-type': 'application/json', 'x-pworker-purpose': 'test:prompted' }, body: JSON.stringify(b) });
  try {
    const text = 'Ana has 12 apples. How many apples does Ana have?';
    let r = await post('/v1/structure', { model: 'structure', text, entities: { quantity: 'a number', goal: 'what is asked' } });
    assert.equal(r.status, 200);
    assert.equal(r.headers.get('x-pworker-model'), 'c/Q+psm');
    let j = await r.json();
    assert.deepEqual(j.entities.quantity[0], { text: '12 apples', start: 8, end: 17, confidence: null });
    assert.equal(j.entities.goal[0].text, 'How many');
    assert.equal(j.reasks, 1); assert.equal(calls, 2);
    r = await post('/v1/structure', { model: 'structure', text, entities: { quantity: 'a number', goal: 'what is asked' } });
    assert.equal(r.headers.get('x-pworker-cache'), 'hit'); assert.equal(calls, 2);
    r = await post('/v1/fol', { model: 'formalizer', inputs: ['Ana has 12 apples.', 'Does Ana have 12 apples?'], context: { inventory: 'things: ana' } });
    j = await r.json();
    assert.deepEqual(j.results.map((x) => x.candidates), [['Has(ana, 12)'], ['? Has(ana, 12)']]);
    r = await post('/v1/fol', { model: 'formalizer', inputs: [] });
    assert.equal(r.status, 400);
    const recs = recsOf(dataDir);
    assert.ok(recs.some((x) => x.tier === 'structure' && x.endpoint === '/v1/chat/completions' && x.purpose === 'test:prompted'));
  } finally {
    p.server.close(); s.close(); p.server.closeAllConnections?.(); s.closeAllConnections?.();
    rmSync(dataDir, { recursive: true, force: true }); rmSync(dataDir + '-cache', { recursive: true, force: true }); rmSync(dir, { recursive: true, force: true });
  }
});

test('local: servers of one exclusive group never run together (starting one stops and awaits the other)', async () => {
  const { createLocalStarter, stopPeers } = await import('../lib/local.mjs');
  const h = launcherHarness();
  const up = new Set(), events = [];
  const spawn = (cmd, argv) => {
    const port = argv[argv.indexOf('--port') + 1];
    up.add(port); events.push(`spawn ${port}`);
    return h.fake(Number(port), { exitDelayMs: 20, onExit: () => { up.delete(port); events.push(`exit ${port}`); } });
  };
  const fetchImpl = async (url) => ({ ok: up.has(new URL(url).port) });
  const logDir = mkdtempSync(join(tmpdir(), 'pworker-excl-'));
  const st = (n) => ({ bin: FAKE_BIN, gguf: FAKE_GGUF, requireFreeGpu: false, exclusiveGroup: 'moe', logFile: join(logDir, `${n}.log`) });
  const cfg = { a: { start: st('a') }, b: { start: st('b') } };
  const starters = {};
  for (const [n, port] of [['a', 1901], ['b', 1902]]) starters[n] = createLocalStarter({ name: n, baseUrl: `http://127.0.0.1:${port}`, start: cfg[n].start }, { spawn, fetchImpl, gpuPids: () => [], killImpl: h.killImpl, beforeStart: () => stopPeers(n, starters, cfg) });
  try {
    assert.equal(await starters.a.ensure(), true);
    assert.equal(await starters.b.ensure(), true);
    assert.deepEqual(events, ['spawn 1901', 'exit 1901', 'spawn 1902']);
    assert.equal(starters.a.status().managed, false);
    assert.equal(starters.b.status().managed, true);
    assert.deepEqual(h.signals, [[1901, 'SIGTERM']], 'starting b signals only the peer, with SIGTERM');
    await starters.b.stop();
    assert.deepEqual(h.signals, [[1901, 'SIGTERM'], [1902, 'SIGTERM']]);
    assert.deepEqual(h.fallbackKills, []);
  } finally { rmSync(logDir, { recursive: true, force: true }); }
});

test('prompted tiers: a reply cut by its budget is asked again with four times the budget, and usage is reported', async () => {
  const { serveprompted, parseTemplate } = await import('../lib/prompted.mjs');
  const template = { ...parseTemplate('<<<options>>>\n{"mode": "problem", "maxTokens": 100}\n<<<user>>>\n{{text}} {{labels}} {{relations}}\n'), name: 't' };
  const budgets = [];
  const chat = async (messages, { maxTokens }) => {
    budgets.push(maxTokens);
    return maxTokens < 1600 ? { ok: true, text: '', finish: 'length', usage: { output_tokens: maxTokens, reasoning_chars: 10 } }
      : { ok: true, text: '{"spans": [{"label": "quantity", "text": "3 cats"}]}', finish: 'stop', usage: { output_tokens: 50, reasoning_chars: 5 } };
  };
  const r = await serveprompted({ path: '/v1/structure', body: { text: 'Tom has 3 cats.', entities: ['quantity'] }, entry: { maxTokensCap: 2000 }, template, chat });
  assert.deepEqual(budgets, [100, 400, 1600]);
  assert.equal(r.body.entities.quantity[0].text, '3 cats');
  assert.deepEqual(r.body.usage, { calls: 3, output_tokens: 550, reasoning_tokens: 0, reasoning_chars: 25, content_chars: 0, budget_retries: 2, cut: 0 });
});

test('prompted tiers: the re-ask keeps the better-formed reply and fills what it lacks from the other (an unreadable reply never wins)', async () => {
  const { serveprompted, parseTemplate } = await import('../lib/prompted.mjs');
  const template = { ...parseTemplate('<<<options>>>\n{"mode": "problem"}\n<<<user>>>\n{{sentences}} {{inventory}}\n<<<again>>>\nFix: {{problems}}\n'), name: 'f' };
  const body = { inputs: ['Ana has 12 apples.', 'Ben has 3.', 'How many in all?'] };
  const serve = (replies) => { let k = 0; return serveprompted({ path: '/v1/fol', body, entry: {}, template, chat: async () => ({ ok: true, text: replies[k++], finish: 'stop' }) }); };
  // The second reply is not JSON: the first, partly valid, is kept.
  let r = await serve(['{"fol": [{"s": 1, "fol": ["Value(ana, 12)"]}, {"s": 2, "fol": ["Value(ben, 3"]}]}', 'I cannot do that.']);
  assert.deepEqual(r.body.results.map((x) => x.candidates), [['Value(ana, 12)'], [], []]);
  assert.equal(r.body.reasks, 1);
  // Both readable: the one with fewer problems is the base, and a sentence it left empty is filled from the other.
  r = await serve(['{"fol": [{"s": 1, "fol": ["Value(ana, 12)"]}, {"s": 2, "fol": ["Value(ben, 3)"]}]}', '{"fol": [{"s": 3, "fol": ["Value(all, add(ana, ben))", "? Ask(all)"]}]}']);
  assert.deepEqual(r.body.results.map((x) => x.candidates), [['Value(ana, 12)'], ['Value(ben, 3)'], ['Value(all, add(ana, ben))\n? Ask(all)']]);
  assert.equal(r.body.unresolved, 0);
});

test('local: the idle stop never stops a server with requests in flight or queued', async () => {
  const { createLocalStarter } = await import('../lib/local.mjs');
  const h = launcherHarness();
  let running = false, busy = true;
  const spawn = () => { running = true; return h.fake(1, { onExit: () => { running = false; } }); };
  const terms = () => h.signals.filter(([, sig]) => sig === 'SIGTERM').length;
  const logDir = mkdtempSync(join(tmpdir(), 'pworker-idle-'));
  const st = createLocalStarter({ name: 'x', baseUrl: 'http://127.0.0.1:1903', start: { bin: FAKE_BIN, gguf: FAKE_GGUF, requireFreeGpu: false, idleStopMs: 30, logFile: join(logDir, 'x.log') } },
    { spawn, fetchImpl: async () => ({ ok: running }), gpuPids: () => [], busy: () => busy, killImpl: h.killImpl });
  try {
    assert.equal(await st.ensure(), true);
    await new Promise((r) => setTimeout(r, 150));
    assert.equal(terms(), 0, 'a busy server is not stopped');
    busy = false;
    await new Promise((r) => setTimeout(r, 150));
    assert.equal(terms(), 1, 'an idle server is stopped');
    assert.deepEqual(h.fallbackKills, []);
  } finally { await st.stop(); rmSync(logDir, { recursive: true, force: true }); }
});

test('launcher safety (negative control): a starter built without killImpl reaches the process.kill guard and nothing is signalled', async () => {
  const { createLocalStarter } = await import('../lib/local.mjs');
  const h = launcherHarness();
  let running = false;
  const child = h.fake(1, { onExit: () => { running = false; } });
  const logDir = mkdtempSync(join(tmpdir(), 'pworker-neg-'));
  const st = createLocalStarter({ name: 'neg', baseUrl: 'http://127.0.0.1:1904', start: { bin: FAKE_BIN, gguf: FAKE_GGUF, requireFreeGpu: false, logFile: join(logDir, 'neg.log') } },
    { spawn: () => { running = true; return child; }, fetchImpl: async () => ({ ok: running }), gpuPids: () => [] });
  try {
    assert.equal(await st.ensure(), true);
    const stopping = st.stop();
    assert.deepEqual(guardHits, [[1, 'SIGTERM']], 'the default process.kill was reached and blocked');
    assert.deepEqual(h.signals, [], 'no signal was recorded through an injected killImpl');
    assert.deepEqual(h.fallbackKills, [[1, 'SIGTERM']], 'the launcher fell back to the fake child handle');
    child.deliver('SIGTERM');
    await stopping;
  } finally { guardHits.length = 0; rmSync(logDir, { recursive: true, force: true }); }
});

test('plan limits: a background share below one call still holds the limit', async () => {
  const { limitWait } = await import('../lib/plan.mjs');
  const now = Date.now();
  const w = limitWait({ upstream: 'u', limits: [{ name: 'pm', unit: 'calls', window: '60s', max: 1 }], records: [{ upstream: 'u', t: now - 1000, status: 200 }], models: {}, now, job: {}, active: [], share: 0.7 });
  assert.ok(w.wait > 50_000, `must wait for the window, got ${w.wait}`);
});
