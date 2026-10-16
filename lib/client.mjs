// The Ploinky Workers library: a client of the Ploinky Workers server (`pworker serve`). Every model call of a program goes through it, by tier
// name, tagged with a purpose (required) and optionally a run id. Outside the server it talks HTTP to the server's URL; inside the
// server (declared task phases) it is given the in-process transport, so the same code runs in both places.
//
//   const ta = createPworkerClient({purpose: 'job:my-batch'});
//   const r = await ta.chat({tier: 'small', messages: [{role: 'user', content: 'Hello'}]});   // {ok, text, finish, cut, usage, served, ...}
//   const j = await ta.json({tier: 'good', prompt: 'Return {"a": 1}'});                       // {..., json}
//   await ta.role('structure').structure({text, entities});                                   // prompted JSON tiers
//   await ta.task({begin: {tier: null, code: "this.end(this.input)"}}, {input: "hello"});
//   await ta.stats();  await ta.models();
import { spawn } from 'node:child_process';
import {setTimeout as delay} from 'node:timers/promises';
import { openSync, closeSync, mkdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {validateTask} from './pworker/task.mjs';
import { DEFAULT_URL, pworkerHome, readServerRecord } from './settings.mjs';
import { httpFetch } from './http-fetch.mjs';

const BIN = resolve(dirname(fileURLToPath(import.meta.url)), '..', 'bin', 'pworker.mjs');
const H = { purpose: 'x-pworker-purpose', run: 'x-pworker-run', cache: 'x-pworker-cache', noFallback: 'x-pworker-no-fallback', priority: 'x-pworker-priority', client: 'x-client-name' };
export const PRIORITIES = Object.freeze(['interactive', 'normal', 'background']);
export const CACHE_MODES = Object.freeze(['use', 'strict', 'record', 'off']);
/** How long an autostarted server is waited for: a server that follows another of this home waits up to 45 s for the predecessor's local models (host.mjs OWNER_WAIT_MS) and then reaps, so this is longer than that. */
export const AUTOSTART_WAIT_MS = 75_000;

/** Text of a reply without a thinking block (closed or left open by a cut reply). */
export const stripThinking = (text) => String(text ?? '').replace(/<think>[\s\S]*?<\/think>/g, '').replace(/<think>[\s\S]*$/g, '').trim();

/** The JSON object in a reply (fences and a thinking block stripped), or null. */
export function jsonOf(text) {
  const s = stripThinking(text).replace(/```(?:json)?/g, '');
  const a = s.indexOf('{'), b = s.lastIndexOf('}');
  if (a < 0 || b <= a) return null;
  try { return JSON.parse(s.slice(a, b + 1)); } catch { return null; }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
// A response's body as text (a test double may offer only json()).
const textOf = async (res) => (typeof res.text === 'function' ? res.text() : JSON.stringify(await res.json()));
const num = (h, k) => { const v = h.get(k); const n = Number(v); return v != null && Number.isFinite(n) ? n : null; };

export class PworkerUnavailable extends Error {
  constructor(url, cause) {
    super(`Pworker proxy not reachable at ${url} (${cause}); start it with: pworker start`);
    this.code = 'pworker_unavailable';
  }
}

/**
 * `url`: the server (default PWORKER_URL, else http://127.0.0.1:18080); `fetchImpl`: the transport (tests and the server's own
 * task phases pass theirs); `purpose`: required tag of every call (`chat`, `formalize`, `answer-*`, `job:<name>`, `review:<run>`,
 * `pworker:<name>`, `run:<id>`, `test:<name>`); `run`: a run id whose registered budget the server enforces; `client`: the name in the server's log;
 * `cache`: the default cache mode (use | strict | record | off; the server's default otherwise); `priority`: interactive | normal |
 * background (background work runs only when nothing else waits and the plan keeps its headroom; held, never refused); `autostart`: start the server when it
 * is not running (on for a local URL with the default transport; off under node --test, with an injected transport, or PWORKER_AUTOSTART=0).
 */
export function createPworkerClient({ url = null, fetchImpl = null, purpose, run = null, client = null, cache = null, priority = null, token = null, autostart = null, config = null, env = process.env } = {}) {
  if (!purpose || typeof purpose !== 'string') throw new TypeError('createPworkerClient: a purpose tag is required (for example "job:<name>", "chat", "test:<name>")');
  // Without an explicit URL or port, the proxy recorded for this home is used (one proxy per home: its limits are shared by all callers).
  const recorded = url == null && !env.PWORKER_URL && !env.PWORKER_PORT && !fetchImpl ? readServerRecord(pworkerHome(env)) : null;
  const recordedUrl = recorded ? `http://${!recorded.host || recorded.host === '0.0.0.0' || recorded.host === '::' ? '127.0.0.1' : recorded.host.includes(':') ? `[${recorded.host}]` : recorded.host}:${recorded.port}` : null;
  const base = String(url ?? env.PWORKER_URL ?? (env.PWORKER_PORT ? `http://127.0.0.1:${env.PWORKER_PORT}` : recordedUrl ?? DEFAULT_URL)).replace(/\/v1\/?$/, '').replace(/\/+$/, '');
  // Under node --test a client without an injected transport or an explicit server never reaches the machine's real server: a test
  // must not call a model (tests inject a fake transport or point `url` at a stub).
  const isolated = !fetchImpl && !url && env.NODE_TEST_CONTEXT && !env.PWORKER_URL;
  const transport = isolated ? async () => { throw Object.assign(new TypeError('fetch failed'), { cause: { code: 'ECONNREFUSED (a test without an injected Ploinky Workers transport)' } }); }
    : fetchImpl ?? httpFetch; // no fixed 300 s header timeout: a long model call ends only at its own timeout
  // Auto-start (owner, 2026-10-03): a client on this machine starts the server when none answers, unless it brought its own transport
  // (tests, the server's own task phases), runs under node --test, or PWORKER_AUTOSTART=0.
  const local = /^http:\/\/(127\.0\.0\.1|localhost)(:\d+)?$/.test(base);
  const auto = autostart ?? (env.PWORKER_AUTOSTART === '1' || (env.PWORKER_AUTOSTART !== '0' && !fetchImpl && local && !env.NODE_TEST_CONTEXT));
  const auth = token ?? env.PWORKER_TOKEN ?? null;
  let starting = null;

  // A cache mode is checked when the client is made and when a call names one: a wrong mode is a programming error, thrown at once.
  const checkCache = (c) => { if (c != null && !CACHE_MODES.includes(c)) throw new TypeError(`cache mode must be one of ${CACHE_MODES.join(', ')}`); };
  checkCache(cache);
  const tags = (o = {}) => {
    const h = { 'content-type': 'application/json', [H.purpose]: o.purpose ?? purpose };
    const r = o.run === undefined ? run : o.run;
    if (r) h[H.run] = r;
    const c = o.cache ?? cache;
    if (c) { if (!CACHE_MODES.includes(c)) throw new TypeError(`cache mode must be one of ${CACHE_MODES.join(', ')}`); h[H.cache] = c; }
    if (o.noFallback) h[H.noFallback] = '1';
    if (Number.isSafeInteger(o.batchSize) && o.batchSize > 1) h['x-pworker-batch-size'] = String(o.batchSize); // recorded: requests saved by batching
    if(o.waitForCapacity){
      h['x-pworker-wait-for-capacity']='1';
      if(o.timeoutMs!=null){
        if(!Number.isSafeInteger(o.timeoutMs)||o.timeoutMs<1)throw new TypeError('timeoutMs must be positive');
        h['x-pworker-upstream-timeout-ms']=String(o.timeoutMs);
      }
    }
    const p = o.priority ?? priority;
    if (p) { if (!PRIORITIES.includes(p)) throw new TypeError(`priority must be one of ${PRIORITIES.join(', ')}`); h[H.priority] = p; }
    if (o.client ?? client) h[H.client] = String(o.client ?? client).slice(0, 40);
    if (auth) h.authorization = `Bearer ${auth}`;
    return { ...h, ...(o.headers ?? {}) };
  };

  async function startServer() {
    const port = new URL(base).port || '18080';
    const logDir = join(pworkerHome(env), 'logs');
    mkdirSync(logDir, { recursive: true });
    const fd = openSync(join(logDir, `serve-${port}.log`), 'a');
    const child = spawn(process.execPath, [BIN, 'serve', '--port', port, ...(config ? ['--config', config] : [])], { detached: true, stdio: ['ignore', fd, fd] });
    child.unref(); closeSync(fd);
    for (let i = 0; i < AUTOSTART_WAIT_MS / 500; i++) {
      await sleep(500);
      try { if ((await transport(`${base}/health`, { signal: AbortSignal.timeout(1000) })).ok) return true; } catch { /* not yet */ }
    }
    return false;
  }

  /** A raw request to the server (path relative to the server root); retries once after starting the server when autostart is on. */
  async function request(path, { method = 'POST', body = undefined, headers = {}, timeoutMs = 600_000, signal = undefined, tagsOf = {} } = {}) {
    const init = () => ({ method, headers: { ...tags(tagsOf), ...headers }, body: body === undefined ? undefined : typeof body === 'string' ? body : JSON.stringify(body), signal: timeoutMs==null?signal:signal ? AbortSignal.any([signal, AbortSignal.timeout(timeoutMs)]) : AbortSignal.timeout(timeoutMs) });
    try { return await transport(`${base}${path}`, init()); }
    catch (e) {
      // A kept-alive socket the server closed while it was idle fails at once with a reset: the request never reached the server, so
      // it is sent again once on a fresh socket (before 2026-10-03 such a call failed as "unavailable" when autostart was off).
      if (/ECONNRESET|EPIPE|socket hang up/i.test(`${e?.cause?.code ?? ''} ${e?.cause?.message ?? ''} ${e?.message ?? ''}`) && e?.name !== 'TimeoutError' && e?.name !== 'AbortError') {
        try { await sleep(100); return await transport(`${base}${path}`, init()); } catch (again) { e = again; }
      }
      const refused = /ECONNREFUSED|fetch failed|ECONNRESET|socket/i.test(`${e?.cause?.code ?? ''} ${e?.message ?? ''}`) && e?.name !== 'TimeoutError' && e?.name !== 'AbortError';
      if (!refused) throw e;
      if (!auto) throw new PworkerUnavailable(base, e?.cause?.code ?? e.message);
      starting ??= startServer().finally(() => { starting = null; });
      if (!(await starting)) throw new PworkerUnavailable(base, 'autostart failed; see ~/.pworker/logs/serve-*.log');
      return transport(`${base}${path}`, init());
    }
  }
  const getJson = async (path, o = {}) => { const r = await request(path, { method: 'GET', ...o }); const j = await r.json().catch(() => null); if (!r.ok) throw Object.assign(new Error(j?.error?.message ?? `status ${r.status}`), { status: r.status, type: j?.error?.type }); return j; };
  const postJson = async (path, body, o = {}) => { const r = await request(path, { body, ...o }); const j = await r.json().catch(() => null); if (!r.ok) throw Object.assign(new Error(j?.error?.message ?? `status ${r.status}`), { status: r.status, type: j?.error?.type }); return j; };

  /**
   * One chat completion. `tier` (or `upstream` + `model` for a concrete model, calibrations only), `messages` (or `system` + `prompt`),
   * `maxTokens`, `temperature`, `extraBody` (merged into the request body as is: sampling, thinking switches), `cache`, `noFallback`,
   * `retries` (transient failures: unreachable, 429, 5xx), `retryCut` (a reply cut by its token budget is asked again with four times the
   * budget up to `cutCap`, default 32000; a reply still cut is a failure `budget_exhausted`). A cut reply is always marked `cut: true` and
   * never cached by the server. Returns {ok, status, text, raw, reasoning, finish, cut, usage, tier, served, fallback, fallbackReason,
   * cached, credits, usd, ms, body, reason}; never throws on a model failure.
   */
  async function chat(o = {}) {
    checkCache(o.cache);
    const messages = o.messages ?? [...(o.system ? [{ role: 'system', content: o.system }] : []), { role: 'user', content: String(o.prompt ?? '') }];
    const model = o.tier ?? o.model;
    if (!model) throw new TypeError('chat: a tier (or upstream + model) is required');
    const path = o.upstream ? `/u/${encodeURIComponent(o.upstream)}/v1/chat/completions` : '/v1/chat/completions';
    let maxTokens = o.maxTokens ?? o.extraBody?.max_tokens;
    const started = Date.now();
    const total = { calls: 0, budget_retries: 0 };
    for (;;) {
      // extraBody first: it never overrides the budget a retryCut raises, the model or the messages.
      const body = { ...(o.extraBody ?? {}), model, messages, ...(maxTokens != null ? { max_tokens: maxTokens } : {}), ...(o.temperature != null ? { temperature: o.temperature } : {}), ...(o.stream === false ? { stream: false } : {}) };
      const r = await once(path, body, o);
      total.calls += 1;
      if (!(r.ok && r.cut && o.retryCut)) return { ...r, calls: total.calls, budget_retries: total.budget_retries, ms: Date.now() - started };
      const cap = o.cutCap ?? 32000;
      if ((maxTokens ?? 0) >= cap) return { ...r, ok: false, reason: `budget_exhausted: cut at ${maxTokens} tokens`, calls: total.calls, budget_retries: total.budget_retries, ms: Date.now() - started };
      maxTokens = Math.min(cap, (maxTokens ?? 1000) * 4); total.budget_retries += 1;
    }
  }

  async function once(path, body, o) {
    const t0 = Date.now();
    let last = null, failures = 0;
    // A capacity signal (429, 529, 503 with retry-after) is waited for without limit under waitForCapacity; any other failure (unreachable,
    // 5xx such as tier_unavailable, proxy_error, local_unavailable, a bad base URL) is retried a bounded number of times, then reported.
    const maxFailures = o.retries ?? (o.waitForCapacity ? 8 : 0);
    for (let attempt = 0; ; attempt++) {
      if (attempt) {
        const wait=Math.min(30000,(o.retryPauseMs ?? 2000)*attempt);
        o.onDeferred?.({reason:last?.reason??'Temporary upstream failure',retryAt:new Date(Date.now()+wait).toISOString()});
        try{await delay(wait,undefined,{signal:o.signal});}catch(e){return {ok:false,status:0,text:'',cancelled:true,reason:'Request cancelled',ms:Date.now()-t0};}
      }
      let res, raw;
      try { res = await request(path, { body, timeoutMs: o.waitForCapacity?null:o.timeoutMs ?? 600_000, signal: o.signal, tagsOf: o }); raw = await textOf(res); }
      catch (e) {
        if(o.signal?.aborted||e?.name==='AbortError')return {ok:false,status:0,text:'',cancelled:true,reason:'Request cancelled',ms:Date.now()-t0};
        if (e instanceof PworkerUnavailable && !o.waitForCapacity) return { ok: false, status: 0, text: '', reason: e.message, ms: Date.now() - t0 };
        last = { ok: false, status: 0, text: '', reason: e?.name === 'TimeoutError' ? `no answer within ${Math.round((o.timeoutMs ?? 600_000) / 1000)} s` : `unreachable: ${e?.message ?? e}`, ms: Date.now() - t0 };
        if (++failures > maxFailures) return last;
        continue;
      }
      let j = null;
      try { j = JSON.parse(raw); } catch { /* reported below */ }
      const h = res.headers ?? new Headers();
      const g = (k) => h.get(`x-pworker-${k}`);
      const meta = { status: res.status, tier: g('tier'), served: g('model'), fallback: g('fallback'), fallbackReason: g('fallback-reason'), cached: g('cache') === 'hit', credits: num(h, 'x-quota-cost'), cacheKey: g('cache-key') };
      if (!res.ok || !j) {
        last = { ok: false, ...meta, text: '', body: j, error: j?.error ?? null, reason: `status ${res.status}: ${j?.error?.type ? `${j.error.type}: ${j.error.message ?? ''}` : raw.slice(0, 200)}`, ms: Date.now() - t0 };
        // A service-time limit is not a capacity wait: a nonterminating model
        // must not be restarted forever. Authentication/schema errors also fail.
        const capacity = res.status === 429 || res.status === 529 || (res.status === 503 && h.get('retry-after') != null);
        if (capacity && o.waitForCapacity) continue;
        if ((capacity || (res.status >= 500 && j?.error?.type !== 'upstream_timeout')) && ++failures <= maxFailures) continue;
        return last;
      }
      const choice = j.choices?.[0];
      const content = choice ? String(choice.message?.content ?? '') : (j.content || []).map((c) => c.text || '').join('');
      const finish = choice ? choice.finish_reason ?? null : j.stop_reason ?? null;
      const u = j.usage ?? {};
      return { ok: true, ...meta, raw: content, text: stripThinking(content), reasoning: choice?.message?.reasoning_content ?? null, finish, cut: finish === 'length' || finish === 'max_tokens',
        usage: { in: u.prompt_tokens ?? u.input_tokens ?? 0, out: u.completion_tokens ?? u.output_tokens ?? 0, cached: u.prompt_tokens_details?.cached_tokens ?? u.cache_read_input_tokens ?? 0, reasoning: u.completion_tokens_details?.reasoning_tokens ?? 0 },
        usd: u.cost ?? null, body: j, ms: Date.now() - t0 };
    }
  }

  /** A chat whose reply must contain one JSON object: the result has `json` (null when unreadable; then `ok` is false). */
  async function json(o = {}) {
    const r = await chat(o);
    if (!r.ok) return { ...r, json: null };
    const parsed = jsonOf(r.text);
    return parsed ? { ...r, json: parsed } : { ...r, ok: false, json: null, reason: 'the reply holds no JSON object' };
  }

  /** A JSON tier endpoint (prompted roles: /v1/structure, /v1/fol): {ok, body, status, ms, cached, served, reason}. */
  async function jsonTier(path, body, o = {}) {
    const t0 = Date.now();
    try {
      const res = await request(path, { body, timeoutMs: o.timeoutMs ?? 600_000, signal: o.signal, tagsOf: o });
      const text = await textOf(res);
      let j = null;
      try { j = JSON.parse(text); } catch { /* below */ }
      if (!res.ok || !j || j.error) return { ok: false, status: res.status, reason: `${res.status} ${j?.error?.message ?? text.slice(0, 200)}`, ms: Date.now() - t0 };
      return { ok: true, status: res.status, body: j, ms: Date.now() - t0, cached: res.headers?.get?.('x-pworker-cache') === 'hit', served: res.headers?.get?.('x-pworker-model') ?? null };
    } catch (e) { return { ok: false, status: 0, reason: String(e?.message ?? e), ms: Date.now() - t0 }; }
  }

  /** A prompted role (a tier with a role prompt, e.g. `structure`, `formalizer`, `formalizer-good`): its JSON endpoints and chat. */
  const role = (name) => ({
    name,
    structure: (body, o = {}) => jsonTier('/v1/structure', { model: name, ...body }, o),
    fol: (body, o = {}) => jsonTier('/v1/fol', { model: name, ...body }, o),
    chat: (o = {}) => chat({ ...o, tier: name }),
  });

  /** Waits for a declarative phased task and returns its persisted record. */
  async function waitOp(id, { pollSeconds = 20, onLog = null } = {}) {
    let seen = 0;
    for (;;) {
      const op = await getJson(`/v1/ops/${encodeURIComponent(id)}?wait=${pollSeconds}&since=${seen}`, { timeoutMs: (pollSeconds + 30) * 1000 });
      if (onLog) for (const line of op.log ?? []) onLog(line);
      seen = op.log_total ?? seen;
      if (!['running','queued','waiting'].includes(op.status)) return op;
    }
  }
  const startOp = async (path, body, o = {}) => {
    const op = await postJson(path, body, { tagsOf: o });
    return o.wait === false ? op : waitOp(op.id, o);
  };

  const agent = {
    url: base, purpose, run,
    /** A copy with other tags (purpose, run, client, cache). */
    with: (o = {}) => createPworkerClient({ url: base, fetchImpl: transport, purpose: o.purpose ?? purpose, run: o.run === undefined ? run : o.run, client: o.client ?? client, cache: o.cache ?? cache, priority: o.priority ?? priority, token: auth, autostart: auto, config, env }),
    request, chat, json, role,
    structure: (body, o = {}) => role(o.tier ?? 'structure').structure(body, o),
    fol: (body, o = {}) => role(o.tier ?? 'formalizer').fol(body, o),
    health: () => getJson('/health'),
    /** The tiers the server serves: [{id, x_tier: {serves, fallback} | {error}}]. */
    tiers: async () => (await getJson('/health')).tiers ?? [],
    stats: () => getJson('/stats'),
    models: () => getJson('/v1/local'),
    providerModels: (name) => getJson(`/u/${encodeURIComponent(name)}/v1/models`),
    model: (name, action) => postJson(`/v1/local/${encodeURIComponent(name)}/${action}`, {}),
    /** The local-llm engine of the server (`pworker models ...`): its admin routes under /v1/local-llm. Reads answer `available: false` with the refusal when the engine cannot be had. */
    engine: {
      overview: () => getJson('/v1/local-llm'),
      status: (sinceSeq = 0) => getJson(`/v1/local-llm/status?sinceSeq=${Number(sinceSeq) || 0}`),
      preview: (body) => postJson('/v1/local-llm/preview', body),
      run: (body) => postJson('/v1/local-llm/run', body, { timeoutMs: 120_000 }),
      stop: () => postJson('/v1/local-llm/stop', {}, { timeoutMs: 120_000 }),
      cancel: () => postJson('/v1/local-llm/cancel', {}),
      deleteWeights: (body) => postJson('/v1/local-llm/weights/delete', body),
      lookup: (body) => postJson('/v1/local-llm/models/lookup', body),
      add: (body) => postJson('/v1/local-llm/models/add', body),
      update: (body) => postJson('/v1/local-llm/models/update', body),
      remove: (modelId) => postJson('/v1/local-llm/models/remove', { modelId }),
      installRunner: (runnerId, acceptLicence = false) => postJson(`/v1/local-llm/runners/${encodeURIComponent(runnerId)}/install`, { acceptLicence }),
      uninstallRunner: (runnerId) => postJson(`/v1/local-llm/runners/${encodeURIComponent(runnerId)}/uninstall`, {}),
    },
    registerRun: (b) => postJson('/jobs/register', b),
    finishRun: (b) => postJson('/jobs/finish', b),
    /** Registered runs with their budgets and spend ({runs: [{run, job, status, budget, spent: {calls, usd, credits}}], by_job}). */
    jobs: () => getJson('/jobs'),
    /** A JSON phase map, with input state and optional exact working directory. */
    task: (task, o = {}) => {validateTask(task);return startOp('/v1/tasks', { task, input:o.input??{}, currentWorkingDirectory:o.currentWorkingDirectory??null }, o);},
    op: (id) => getJson(`/v1/ops/${encodeURIComponent(id)}`),
    ops: () => getJson('/v1/ops'),
    waitOp,
    cancelTask: id=>postJson(`/v1/tasks/${encodeURIComponent(id)}/cancel`,{}),
  };
  return agent;
}
