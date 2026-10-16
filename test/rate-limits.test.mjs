// Provider rate limits: every attempt is counted when sent, pacing, the full Retry-After, an adaptive rate after a 429 below the configured
// limit, a held slot after an abort, and the invariant that only the proxy core sends model requests to providers.
import './home.mjs';
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { Limiter } from '../lib/limiter.mjs';
import { createCore } from '../lib/core.mjs';
import { inprocFetch } from '../lib/inproc.mjs';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
// Limiter timers do not hold the event loop open (a listening server does): keep it alive while these tests wait.
const keep = setInterval(() => {}, 1000);
test.after(() => clearInterval(keep));

test('even pacing spreads starts instead of bursting up to the window cap', async () => {
  const l = new Limiter({ maxConcurrent: 10, maxPerMinute: 600, pacing: 'even' }); // one start every 100 ms
  const at = [];
  await Promise.all([0, 1, 2, 3].map(() => l.schedule(() => { at.push(Date.now()); })));
  for (let i = 1; i < at.length; i++) assert.ok(at[i] - at[i - 1] >= 90, `gap ${at[i] - at[i - 1]} ms`);
  assert.equal(l.effective().spacingMs, 100);
});

test('the safety margin and the window slack lower what is sent', () => {
  const l = new Limiter({ maxPerMinute: 15, safetyMargin: 0.2, windowSlackMs: 2000 });
  assert.equal(l.effective().maxPerMinute, 12);
  l.seed(Array.from({ length: 12 }, () => Date.now() - 60_500)); // outside 60 s but inside 60 s + slack
  assert.ok(l.estimateWait().wait > 1000);
});

test('a 429 lowers the adaptive rate once per burst; it recovers step by step after cooldowns', async () => {
  const l = new Limiter({ maxConcurrent: 4, maxPerMinute: 20, adaptive: { decrease: 0.5, min: 0.2, cooldownMs: 100, increase: 0.25, gapMs: 1000 } });
  l.seed([Date.now()]);
  const a = l.throttled();
  assert.deepEqual([a.underLimit, a.factor, a.adapted], [true, 0.5, true]);
  assert.equal(l.throttled().adapted, false, 'a second 429 of the same burst does not lower again');
  assert.equal(l.effective().maxPerMinute, 10);
  assert.equal(l.effective().maxConcurrent, 2);
  await sleep(230);
  assert.equal(l.effective().factor, 1);
  const again = new Limiter({ maxPerMinute: 2, adaptive: false });
  again.seed([Date.now(), Date.now(), Date.now()]);
  assert.deepEqual(again.throttled(), { underLimit: false, factor: 1, adapted: false });
});

function proxy(upstream, limits = {}, retry = { max: 3, baseMs: 10, maxWaitMs: 500 }) {
  const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'pworker-rate-'));
  const core = createCore({ config: { defaultUpstream: 'p', upstreams: { p: { baseUrl: 'http://p.test', noKey: true, limits: { maxConcurrent: 2, maxPerMinute: 15, ...limits }, retry } } }, env: {}, dataDir, fetchImpl: async (url, init) => (url.endsWith('/v1/models') ? Response.json({ data: [] }) : upstream(url, init)) });
  const f = inprocFetch(core.handle);
  const ask = (headers = {}, signal) => f('http://pworker.local/v1/chat/completions', { method: 'POST', headers: { 'content-type': 'application/json', 'x-pworker-purpose': 'test:rate', ...headers }, body: JSON.stringify({ model: 'm', messages: [{ role: 'user', content: 'q' }] }), signal });
  return { core, ask, limiter: core.limiters.p };
}

test('a Retry-After beyond the wait budget is never cut short: the 429 is returned and the upstream stays paused', async () => {
  let calls = 0;
  const t = proxy(async () => { calls += 1; return Response.json({ error: 'rate limited' }, { status: 429, headers: { 'retry-after': '2' } }); });
  const r = await t.ask();
  assert.equal(r.status, 429);
  assert.equal(calls, 1, 'no resend before Retry-After');
  assert.ok(t.limiter.pausedUntil - Date.now() > 1500);
  const rec = t.core.monitor.records.find((x) => x.status === 429);
  assert.equal(rec.under_limit, true); assert.equal(rec.sent_60s, 1); assert.ok(rec.t_sent <= rec.t);
  assert.ok(rec.rate_factor < 1);
});

test('every attempt of a request passes the limiter and is counted when sent (429 and 5xx retries included)', async () => {
  let n = 0;
  const t = proxy(async () => { n += 1; if (n === 1) return Response.json({}, { status: 429, headers: { 'retry-after': '0.05' } }); if (n === 2) return Response.json({}, { status: 503 }); return Response.json({ choices: [{ message: { content: 'ok' }, finish_reason: 'stop' }] }); },
    { adaptive: false }, { max: 3, max5xx: 2, baseMs: 10, maxWaitMs: 500 });
  const r = await t.ask();
  assert.equal(r.status, 200); await r.text();
  assert.equal(n, 3);
  assert.equal(t.limiter.sentWithin(60_000), 3);
  assert.equal(t.core.monitor.records.filter((r) => r.t_sent).length, 3);
});

test('an attempt aborted after it was sent keeps its concurrency slot for abortHoldMs', async () => {
  const t = proxy((url, init) => new Promise((resolve, reject) => init.signal.addEventListener('abort', () => reject(init.signal.reason))), { maxConcurrent: 1, abortHoldMs: 300, adaptive: false });
  const ctl = new AbortController();
  const pending = t.ask({}, ctl.signal).catch(() => null);
  await sleep(30);
  assert.equal(t.limiter.active, 1);
  ctl.abort();
  await pending; await sleep(50);
  assert.equal(t.limiter.active, 1, 'held after the abort');
  assert.equal(t.core.monitor.records.at(-1).status, 499);
  await sleep(350);
  assert.equal(t.limiter.active, 0);
});

test('invariant: only the proxy core sends model requests to providers', () => {
  const root = path.resolve('.');
  const files = ['bin', 'lib', 'lib/pworker'].flatMap((d) => fs.readdirSync(path.join(root, d)).filter((f) => f.endsWith('.mjs')).map((f) => path.join(d, f)));
  const senders = [];
  for (const file of files) {
    const text = fs.readFileSync(path.join(root, file), 'utf8');
    // A provider request is a fetch to an upstream's baseUrl; only GET catalog and health probes may do that outside the core's forward().
    for (const line of text.split('\n')) if (/baseUrl/.test(line) && /fetch\w*\(/.test(line) && /method:\s*'POST'/.test(line)) senders.push(file);
  }
  assert.deepEqual(senders, ['lib/core.mjs']);
  const core = fs.readFileSync(path.join(root, 'lib/core.mjs'), 'utf8');
  // The one send site serves a provider (its fetch and base URL) and the local-llm engine (its fetch and the target's base URL, over its Unix socket).
  assert.equal(core.match(/fetchVia\(\(target \? target\.baseUrl : up\.baseUrl\) \+ upstreamPath/g)?.length, 1, 'one send site, inside forward()');
  assert.equal(core.match(/const fetchVia = target \? engineFetchImpl : fetchImpl;/g)?.length, 1, 'the send site chooses between the provider fetch and the engine fetch only');
  assert.match(core.slice(core.indexOf('async function forward('), core.indexOf('fetchVia((target ? target.baseUrl : up.baseUrl) + upstreamPath')), /limiter\.schedule\(/, 'the send site runs inside limiter.schedule');
  // Clients (CLI, detached workers, the library) reach models only through the proxy's HTTP API.
  for (const file of ['lib/client.mjs', 'bin/pworker.mjs', 'lib/pworker/task.mjs']) assert.doesNotMatch(fs.readFileSync(path.join(root, file), 'utf8'), /createCore|createProxy/, file);
});

test('metrics: per provider/model and hour, 429s under the configured limit, retries, cache hits, batching savings; the CLI reads the log', async () => {
  const { summarize, formatSummary, readRecords } = await import('../lib/metrics.mjs');
  const { execFile } = await import('node:child_process');
  const { promisify } = await import('node:util');
  const now = Date.now(), t = (s) => now - 600_000 + s * 1000;
  const recs = [
    { upstream: 'openference', model: 'M', tier: 'small', t: t(0), t_sent: t(0), status: 200, attempt: 1, in_tokens: 10, out_tokens: 5, queue_wait_ms: 0 },
    { upstream: 'openference', model: 'M', tier: 'small', t: t(1), t_sent: t(1), status: 429, attempt: 1, queue_wait_ms: 100 },
    { upstream: 'openference', model: 'M', tier: 'small', t: t(3), t_sent: t(3), status: 429, attempt: 2, under_limit: false },
    { upstream: 'openference', model: 'M', tier: 'small', t: t(6), t_sent: t(5), status: 200, attempt: 3, batch_size: 4, in_tokens: 40, out_tokens: 20 },
    { upstream: 'openference', model: 'M', t: t(7), t_sent: t(7), status: 504, timeout: true, attempt: 1 },
    { upstream: 'cache', model: 'small', served: 'openference/M', tier: 'small', t: t(8), status: 200 },
    { upstream: 'openference', model: 'M', t: t(9), status: 402, not_sent: true },
    { upstream: 'other', model: 'X', t: t(10), t_sent: t(10), status: 200, attempt: 1 },
  ];
  const s = summarize(recs, { since: now - 3600_000, now, provider: 'openference', limits: { openference: { maxPerMinute: 15 } }, bucket: 'hour' });
  assert.equal(s.groups.length, 1);
  const g = s.groups[0];
  assert.deepEqual([g.sent, g.ok, g.r429, g.r429_under_limit, g.errors, g.retries, g.timeouts, g.cache_hits, g.batch_saved, g.in_tokens, g.peak_per_minute], [5, 2, 2, 1, 2, 2, 1, 1, 3, 50, 5]);
  assert.deepEqual(g.tiers, ['small']);
  assert.ok(s.buckets.length >= 1);
  const text = formatSummary(s, { limits: { openference: { maxPerMinute: 15 } } });
  assert.match(text, /openference\/M/); assert.match(text, /429<lim/); assert.match(text, /1 of 2 429 responses/);
  // The CLI prints the same from the request log of the home, without a proxy.
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'pworker-stats-'));
  try {
    fs.mkdirSync(path.join(home, 'data'), { recursive: true });
    for (const r of recs) fs.appendFileSync(path.join(home, 'data', `requests-${new Date(r.t).toISOString().slice(0, 10)}.jsonl`), JSON.stringify(r) + '\n');
    assert.equal(readRecords(path.join(home, 'data'), now - 3600_000).length, recs.length);
    const out = await promisify(execFile)(process.execPath, [path.resolve('bin/pworker.mjs'), 'stats', '--since', '1h', '--provider', 'openference', '--json'], { env: { ...process.env, PWORKER_HOME: home } });
    const j = JSON.parse(out.stdout);
    assert.equal(j.groups[0].sent, 5); assert.equal(j.total.r429_under_limit, 1);
    const table = await promisify(execFile)(process.execPath, [path.resolve('bin/pworker.mjs'), 'stats', '--since', '1h'], { env: { ...process.env, PWORKER_HOME: home } });
    assert.match(table.stdout, /other\/X/);
  } finally { fs.rmSync(home, { recursive: true, force: true }); }
});
