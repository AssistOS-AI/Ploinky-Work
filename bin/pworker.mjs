#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline/promises';
import { stdin, stdout } from 'node:process';
import { spawn } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { createPworkerClient } from '../lib/client.mjs';
import { ensureUserHome, loadLayers } from '../lib/config.mjs';
import { pworkerHome } from '../lib/settings.mjs';
import { Pworker } from '../lib/pworker/task.mjs';
import { jobStore } from '../lib/pworker/jobs.mjs';
import { selectMenu, promptChat, promptField } from '../lib/pworker/menu.mjs';
import { connectedProviders, fetchModelCatalog, formatModelLabel, isModelEligibleForTier, modelExclusionReason, normalizeModelCatalog, providerSpec, sortModelCatalog } from '../lib/pworker/providers.mjs';
import { resolveUpstream } from '../lib/settings.mjs';
import { HELP } from '../lib/pworker/help.mjs';
import { liveServer, readServerRecord } from '../lib/server.mjs';
import { fileURLToPath } from 'node:url';
import { readUserConfig as adminReadUserConfig, saveUserConfig as adminSaveUserConfig, saveModelLimits as adminSaveModelLimits, proxyPid, startProxy as adminStartProxy, stopProxyResult as adminStopProxyResult, restartProxy as adminRestartProxy } from '../lib/pworker/admin.mjs';
import { readRecords, summarize, formatSummary, parseSince } from '../lib/metrics.mjs';
import { isEngineCommand, runEngineVerb } from '../lib/pworker/models-cli.mjs';

const args = process.argv.slice(2);
const command = args[0];
const home = pworkerHome();
const configFile = path.join(home, 'config.json');
const queueFile = path.join(home, 'queue.json');
const option = (name, fallback = null) => { const i = args.indexOf(`--${name}`); return i < 0 ? fallback : args[i + 1]; };
const json = (v) => console.log(JSON.stringify(v, null, 2));
const client = () => createPworkerClient({ purpose: option('purpose', 'pworker:cli'), client: 'pworker' });
const sleep = (n) => new Promise((resolve) => setTimeout(resolve, n));

// The persistent queue: every change happens under a lock folder, and a flush first renames the queue to a snapshot of its own, so
// entries queued while a flush runs are never lost and two flushes never run the same entry.
const queueLock = path.join(home, 'queue.lock');
async function withQueueLock(fn) {
  for (let i = 0; ; i++) {
    try { fs.mkdirSync(queueLock); break; }
    catch (error) {
      if (error.code !== 'EEXIST') throw error;
      try { if (Date.now() - fs.statSync(queueLock).mtimeMs > 30_000) { fs.rmdirSync(queueLock); continue; } } catch {}
      if (i > 400) throw new Error(`The queue is locked by another pworker command (${queueLock})`);
      await sleep(25);
    }
  }
  try { return await fn(); } finally { try { fs.rmdirSync(queueLock); } catch {} }
}
const readQueue = (file = queueFile) => { try { return JSON.parse(fs.readFileSync(file, 'utf8')); } catch (error) { if (error.code === 'ENOENT') return []; throw error; } };
function writeQueue(entries) { const tmp = `${queueFile}.${process.pid}.tmp`; fs.writeFileSync(tmp, JSON.stringify(entries, null, 2) + '\n', { mode: 0o600 }); fs.renameSync(tmp, queueFile); }
const appendQueue = (entries) => withQueueLock(() => { const all = [...readQueue(), ...entries]; writeQueue(all); return all.length; });
/** Moves the whole queue to a snapshot file owned by this flush and returns {entries, snapshot}. */
const takeQueue = () => withQueueLock(() => {
  if (!fs.existsSync(queueFile)) return { entries: [], snapshot: null };
  const snapshot = path.join(home, `queue-flush-${process.pid}-${randomUUID()}.json`);
  fs.renameSync(queueFile, snapshot);
  return { entries: readQueue(snapshot), snapshot };
});
// Text that names a task file (an existing path, or a single word ending in .json/.md/.markdown/.mjs) is stored as an absolute path at queue time.
const isTaskFile = (request) => /\.(mjs|json|md|markdown)$/.test(request) && (fs.existsSync(request) || !/\s/.test(request.trim()));
const KNOWN_COMMANDS = new Set(['help', 'status', 'serve', 'start', 'stop', 'stats', 'models', 'provider', 'tier', 'queue', 'run', 'flush', '_execute-batch']);

// `pworker stats`: request metrics read from the request log of this home (no running proxy needed).
function printStats({ since = option('since', '24h'), provider = option('provider'), asJson = args.includes('--json'), by = option('by') } = {}) {
  const config = loadLayers().config;
  const span = parseSince(since), now = Date.now();
  const limits = Object.fromEntries(Object.entries(config.providers ?? config.upstreams ?? {}).filter(([n]) => !n.startsWith('_')).map(([n, p]) => [n, p?.limits ?? {}]));
  const dataDir = config.dataDir ?? path.join(home, 'data');
  const bucket = by === 'day' || by === 'hour' ? by : span > 2 * 86400_000 ? 'day' : 'hour';
  const summary = summarize(readRecords(dataDir, now - span), { since: now - span, now, provider, limits, bucket });
  if (asJson) json(summary); else process.stdout.write(formatSummary(summary, { limits }));
}

const readUserConfig = () => adminReadUserConfig();
const saveUserConfig = (c) => adminSaveUserConfig(c);
// The proxy of this home: its recorded process exists and answers /health on the recorded port (lib/server.mjs). No process-name
// matching, so it works with npm link, other launchers and on every platform. The process control lives in lib/pworker/admin.mjs.
const runningPid = () => proxyPid();
const startProxy = () => adminStartProxy();
/** What `pworker stop` says: the proxy stopped, was not running, or is still draining (it exits only after its local model is stopped). */
const stopMessage = async () => {
  const result = await adminStopProxyResult();
  return result.state === 'stopped' ? 'Proxy stopped.' : result.state === 'still-running' ? `The proxy (pid ${result.pid}) was told to stop but is still running after 45 s; it exits once its local model is stopped. Run pworker stop again later.` : 'The proxy is not running.';
};
const restartProxy = () => adminRestartProxy();

function publicJob(job) {
  if (!job) return null;
  const { id, status, phase, steps, currentWorkingDirectory, createdAt, updatedAt, result, error,waitingReason,retryAt } = job;
  return { id, status, phase, steps, currentWorkingDirectory, createdAt, updatedAt, result, error,waitingReason,retryAt };
}

function startDetachedTasks(items) {
  const store = jobStore(home);
  const prepared = items.map(({ request, input, currentWorkingDirectory }) => {
    const supplied = currentWorkingDirectory ?? (input && typeof input === 'object' && !Array.isArray(input) ? input.currentWorkingDirectory : null);
    const cwd = supplied ? fs.realpathSync(path.resolve(supplied)) : null;
    if (cwd && !fs.statSync(cwd).isDirectory()) throw new Error(`Working directory is not a directory: ${cwd}`);
    return { request, input, currentWorkingDirectory: cwd };
  });
  const jobs = prepared.map((item) => store.create(item));
  const batchId = randomUUID();
  const manifest = path.join(store.dir, `batch-${batchId}.json`);
  fs.writeFileSync(manifest, JSON.stringify(jobs.map((job) => job.id)), { mode: 0o600 });
  const log = fs.openSync(path.join(home, 'logs', `batch-${batchId}.log`), 'a', 0o600);
  try {
    const child = spawn(process.execPath, [fileURLToPath(import.meta.url), '_execute-batch', batchId, '--purpose', option('purpose', 'pworker:cli')], { cwd: process.cwd(), detached: true, stdio: ['ignore', log, log] });
    if (!child.pid) throw new Error('Could not start the task process');
    for (const job of jobs) store.update(job.id, { pid: child.pid });
    child.unref();
  } catch (error) {
    for (const job of jobs) store.update(job.id, { status: 'failed', error: error.message });
    throw error;
  } finally { fs.closeSync(log); }
  return jobs.map((job) => publicJob(store.read(job.id)));
}

async function executeDetachedBatch(batchId) {
  const store = jobStore(home);
  if (!/^[a-f0-9-]{36}$/.test(batchId)) throw new Error('Invalid batch ID');
  const manifest = path.join(store.dir, `batch-${batchId}.json`);
  const ids = JSON.parse(fs.readFileSync(manifest, 'utf8'));
  try {
    const worker = new Pworker({ client: client(), config: loadLayers().config, onProgress: (event) => store.update(event.id, event) });
    for (const id of ids) {
      const job = store.read(id);
      if (!job) continue;
      store.update(id, { pid: process.pid, status: 'compiling' });
      try { await worker.enqueueRequest(job.request, job.input, { id, currentWorkingDirectory: job.currentWorkingDirectory }); }
      catch (error) { store.update(id, { status: 'failed', phase: null, error: error.message }); }
    }
    if (worker.pending.length) await worker.flush();
  } catch (error) {
    for (const id of ids) if (!['completed', 'failed'].includes(store.read(id)?.status)) store.update(id, { status: 'failed', phase: null, error: error.message });
  } finally { try { fs.unlinkSync(manifest); } catch {} }
}

function saveProvider(name, spec, key) {
  const c = readUserConfig();
  c.providers ??= {};
  c.providers[name] = spec;
  saveUserConfig(c);
  if (key) fs.writeFileSync(path.join(home, 'keys', spec.envFile ?? `${name}.env`), `${spec.keyVar}=${JSON.stringify(key)}\n`, { mode: 0o600 });
}
const saveModelLimits = (model) => adminSaveModelLimits(model);

function setTier(tier, entries, batch) {
  const c = readUserConfig();
  c.tiers ??= {};
  c.tiers[tier] = entries.map(({ provider, model }) => ({ upstream: provider, model }));
  c.batching ??= {};
  if (batch !== undefined) c.batching[tier] = { enabled: batch };
  saveUserConfig(c);
}
const ui = (rl) => ({ input: stdin, output: stdout, rl });
const note = (message) => stdout.write(`\n${message}\n`);

async function localStates() {
  try { return await client().models(); }
  catch { return {}; }
}

async function modelsFromProxy(provider) {
  const response = await client().providerModels(provider);
  const spec = loadLayers().config.providers?.[provider] ?? {};
  const models = normalizeModelCatalog(provider, spec, response.data ?? []);
  if (!models.length) throw new Error(`${provider} returned no selectable models`);
  return models;
}

const tierEntries = (config, tier) => Array.isArray(config.tiers?.[tier]) ? config.tiers[tier].map((entry) => ({ provider: entry.upstream, model: entry.model })) : [];
const chainSummary = (entries) => entries.map((entry, index) => `${index + 1}. ${entry.provider}/${entry.model}`).join('\n');

async function tierModelChoices(available, preferred = null) {
  const providers = preferred && available.includes(preferred) ? [preferred, ...available.filter((name) => name !== preferred)] : available;
  const catalogues = await Promise.all(providers.map(async (provider) => {
    try { return { provider, models: await modelsFromProxy(provider) }; }
    catch (error) { return { provider, models: [], error }; }
  }));
  const choices = catalogues.flatMap(({ models }) => sortModelCatalog(models, 'recommended').filter(isModelEligibleForTier));
  return { choices, catalogues, unavailable: catalogues.filter(({ error }) => error).map(({ provider, error }) => `${provider}: ${error.message}`) };
}

async function connectProvider(screen) {
  for (;;) {
    const config = loadLayers().config;
    const local = await localStates();
    const choices = Object.entries(config.providers ?? {}).filter(([name]) => !name.startsWith('_')).map(([name, provider]) => {
      if (provider.start) return { label: `${name} · local ${provider.start.alias ?? ''} · ${local[name]?.running ? 'running' : 'stopped'}`, value: `local:${name}` };
      if (provider.kind === 'local-llm') return { label: `${name} · local engine ${local[name]?.modelId ?? ''} · ${local[name]?.running ? 'running' : 'stopped'}`, value: `local:${name}` };
      const key = resolveUpstream(name, provider).key;
      return { label: `${name} · ${key || provider.noKey ? 'connected' : 'API key needed'}`, value: `api:${name}` };
    });
    choices.push({ label: 'Add OpenAI-compatible endpoint', value: 'custom' });
    const selected = await selectMenu(screen, 'Log in / connect provider', choices);
    if (!selected) return null;
    try {
      if (selected.startsWith('local:')) {
        const name = selected.slice(6);
        if (!local[name]?.running && config.providers[name]?.kind === 'local-llm') throw new Error(`${name} is the local-llm engine; start a model with: pworker models run <id>`);
        if (!local[name]?.running) {
          note(`Starting ${name}...`);
          const result = await client().model(name, 'start');
          if (!result.ok) throw new Error(result.status?.last_refusal?.reason ?? `Could not start ${name}`);
        }
        const models = await modelsFromProxy(name);
        note(`Connected to ${name} (${models.length} models).`);
        return { name, models };
      }
      if (selected === 'custom') {
        const name = await promptField(screen, 'Provider name');
        if (name == null) continue;
        if (config.providers?.[name]?.start) throw new Error('A managed local provider cannot be replaced here');
        const endpoint = await promptField(screen, 'Base endpoint URL');
        if (endpoint == null) continue;
        const key = await promptField(screen, 'API key (leave empty for a keyless local endpoint)', { secret: true, allowEmpty: true });
        if (key == null) continue;
        const rpm = await promptField(screen, 'Maximum requests per minute', { initial: '60' });
        if (rpm == null) continue;
        const spec = providerSpec(name, endpoint, key, rpm, config.providers?.[name] ?? {});
        note('Checking the provider model list...');
        const models = await fetchModelCatalog(spec, { key });
        saveProvider(name, spec, key);
        await restartProxy();
        note(`Connected to ${name} (${models.length} models).`);
        return { name, models };
      }
      const name = selected.slice(4), provider = config.providers[name];
      const resolved = resolveUpstream(name, provider);
      let key = resolved.key;
      if (key || provider.noKey) {
        const action = await selectMenu(screen, `${name} credentials`, [
          { label: key ? 'Use configured API key' : 'Use keyless endpoint', value: 'use' },
          ...(!process.env[provider.keyVar] ? [{ label: 'Replace API key', value: 'replace' }] : []),
        ]);
        if (!action) continue;
        if (action === 'replace') key = await promptField(screen, `${name} API key`, { secret: true });
      } else key = await promptField(screen, `${name} API key`, { secret: true });
      if (key == null) continue;
      note('Checking the provider model list...');
      const models = await fetchModelCatalog(provider, { key });
      if (key !== resolved.key) {
        fs.writeFileSync(path.join(home, 'keys', provider.envFile ?? `${name}.env`), `${provider.keyVar}=${JSON.stringify(key)}\n`, { mode: 0o600 });
        await restartProxy();
      }
      note(`Connected to ${name} (${models.length} models).`);
      return { name, models };
    } catch (error) { note(`Connection failed: ${error.message}`); }
  }
}

async function configureTier(screen, preferred = null) {
  const config = loadLayers().config;
  const local = await localStates();
  const available = connectedProviders(config, local);
  if (!available.length) { note('Connect an API provider or start a local model before configuring a tier.'); return; }
  const names = Object.entries(config.tiers ?? {}).filter(([name, value]) => !name.startsWith('_') && name !== 'maxWaitMs' && Array.isArray(value)).map(([name]) => name);
  const tier = await selectMenu(screen, 'Choose a tier', names.map((name) => {
    const entries = tierEntries(config, name);
    return { label: `${name}${entries.length ? ` · ${entries.length} model${entries.length === 1 ? '' : 's'} · ${entries[0].provider}/${entries[0].model}` : ' · not configured'}`, value: name };
  }));
  if (!tier) return;
  note('Loading the usable text models from connected providers...');
  const { choices, catalogues, unavailable } = await tierModelChoices(available, preferred);
  if (unavailable.length) note(`Catalog unavailable: ${unavailable.join('; ')}`);
  if (!choices.length) { note('No connected provider returned a usable text model.'); return; }
  const current = tierEntries(config, tier);
  const selected = await selectMenu(screen, `Choose the primary model for ${tier} · ${choices.length} available\nSelected model becomes primary; existing usable models remain fallbacks.`, choices.map((model) => ({ label: formatModelLabel(model), value: model })), { searchable: true });
  if (!selected) return;
  const known = new Map(catalogues.flatMap(({ models }) => models).map((model) => [`${model.provider}\u0000${model.id}`, model]));
  const dropped = current.filter((entry) => {
    const model = known.get(`${entry.provider}\u0000${entry.model}`);
    return model && !isModelEligibleForTier(model);
  });
  const entries = [{ provider: selected.provider, model: selected.id }, ...current.filter((entry) => entry.provider !== selected.provider || entry.model !== selected.id).filter((entry) => !dropped.includes(entry))];
  if (dropped.length) note(`Removed unavailable fallback${dropped.length === 1 ? '' : 's'}: ${dropped.map((entry) => `${entry.provider}/${entry.model}`).join(', ')}`);
  note(`${tier} priority order:\n${chainSummary(entries)}`);
  const confirmation = await selectMenu(screen, `Save ${tier}?`, [{ label: 'Save this tier mapping', value: 'save' }]);
  if (confirmation !== 'save') return;
  saveModelLimits(selected);
  setTier(tier, entries);
  await restartProxy();
  note(`Saved ${tier} with ${entries.length} model${entries.length === 1 ? '' : 's'} in priority order.`);
}

async function configurePriceLadder(screen) {
  const config = loadLayers().config;
  const available = connectedProviders(config, await localStates());
  if (!available.length) { note('Connect an API provider or start a local model before creating a price ladder.'); return; }
  note('Loading live catalogs and their published prices...');
  const catalogues = new Map(await Promise.all(available.map(async (provider) => {
    try { return [provider, await modelsFromProxy(provider)]; } catch { return [provider, []]; }
  })));
  const sources = [...catalogues].map(([provider, models]) => {
    const eligible = models.filter(isModelEligibleForTier);
    const priced = eligible.filter((model) => model.prices.input != null || model.prices.output != null).length;
    const excluded = models.length - eligible.length;
    return { provider, models: eligible, priced, excluded };
  }).filter(({ models }) => models.length);
  if (!sources.length) { note('No connected provider returned a selectable model catalog.'); return; }
  const source = await selectMenu(screen, 'Choose the provider used for the suggested price ladder', [
    { label: 'Keep current tiers · do not change anything', value: 'scope:cancel' },
    ...sources.map(({ provider, models, priced, excluded }) => ({ label: `${provider} · ${models.length} usable text models · ${priced} with published prices${excluded ? ` · ${excluded} excluded` : ''}`, value: `provider:${provider}` })),
    ...(sources.length > 1 ? [{ label: `Mix all ${sources.length} providers by published price`, value: 'scope:all' }] : []),
  ]);
  if (!source || source === 'scope:cancel') return;
  const provider = source.startsWith('provider:') ? source.slice('provider:'.length) : null;
  const candidates = source === 'scope:all' ? sources.flatMap(({ models }) => models) : catalogues.get(provider) ?? [];
  const priced = candidates.filter((model) => model.prices.input != null || model.prices.output != null);
  if (!priced.length) { note(`${provider} did not publish token prices. Select models manually or add modelPricing in the user configuration.`); return; }
  const ranked = sortModelCatalog(priced, 'lowest-price');
  const tiers = Object.entries(config.tiers ?? {}).filter(([name, value]) => !name.startsWith('_') && name !== 'maxWaitMs' && Array.isArray(value)).map(([name]) => name);
  const proposal = tiers.map((tier, index) => {
    const model = ranked[Math.round(index * (ranked.length - 1) / Math.max(1, tiers.length - 1))];
    return { tier, provider: model.provider, model: model.id, label: formatModelLabel(model) };
  });
  const sourceLabel = source === 'scope:all' ? 'all selected providers' : provider;
  note(`Suggested price ladder from ${sourceLabel}, based on usable text models with live published prices. Models known to require a credit balance are excluded.\n${proposal.map((entry) => `${entry.tier} → ${entry.label}`).join('\n')}`);
  const review = await selectMenu(screen, 'Review complete. Continue to the final save confirmation?', [
    { label: 'Keep current tiers · do not apply this proposal', value: 'cancel' },
    { label: 'Continue to final confirmation', value: 'continue' },
  ]);
  if (review !== 'continue') return;
  const confirm = await selectMenu(screen, `Apply this ${sourceLabel} price ladder to all tiers?`, [
    { label: 'Keep current tiers · cancel', value: 'cancel' },
    { label: 'Apply the proposed tier mappings', value: 'save' },
  ]);
  if (confirm !== 'save') return;
  const user = readUserConfig();
  user.tiers ??= {};
  for (const entry of proposal) user.tiers[entry.tier] = [{ upstream: entry.provider, model: entry.model }];
  saveUserConfig(user);
  await restartProxy();
  note('Saved the suggested price ladder. You can add provider/model fallbacks to every tier afterwards.');
}

async function manageLocalModels(screen) {
  const local = await localStates();
  const name = await selectMenu(screen, 'Local models', Object.entries(local).map(([id, model]) => ({ label: `${id} · ${model.model ?? 'model'} · ${model.running ? 'running' : 'stopped'}`, value: id })));
  if (!name) return;
  const action = await selectMenu(screen, `${name}`, [
    { label: 'Start model', value: 'start' },
    { label: 'Stop model', value: 'stop' },
  ]);
  if (!action) return;
  const response = await client().model(name, action);
  if (!response.ok) throw new Error(response.status?.last_refusal?.reason ?? `${action} failed`);
  note(`${name}: ${action} completed.`);
}

function taskLabel(job) {
  const phase = job.phase ? ` · phase ${job.phase}` : '';
  const steps = job.steps ? ` · ${job.steps} step${job.steps === 1 ? '' : 's'}` : '';
  const outcome = job.status === 'completed' ? ' · result ready' : job.status === 'failed' ? ' · failed' : '';
  return `${job.status.toUpperCase()} · ${job.id.slice(0, 8)}${phase}${steps}${outcome}`;
}

function taskRecord(job) {
  const rendered = JSON.stringify(publicJob(job), null, 2);
  return rendered.length > 12_000 ? `${rendered.slice(0, 12_000)}\n… task record truncated in this view` : rendered;
}

async function inspectTask(screen, id) {
  const store = jobStore(home);
  for (;;) {
    const job = store.view(id);
    if (!job) { note(`Task ${id} no longer exists.`); return; }
    const action = await selectMenu(screen, `Task ${id.slice(0, 8)} · ${job.status}${job.phase ? ` · phase ${job.phase}` : ''}`, [
      { label: 'Show result, status, and task state', value: 'show' },
      { label: 'Refresh status', value: 'refresh' },
    ]);
    if (!action) return;
    if (action === 'refresh') continue;
    await selectMenu(screen, `Task record\n\n${taskRecord(job)}`, [{ label: 'Back to task', value: 'back' }]);
  }
}

async function composeTasks(screen, currentWorkingDirectory) {
  const transcript = [{ role: 'pworker', text: `New tasks run in the background in ${currentWorkingDirectory}. Write a request and press Enter. The request is also the task's initial input.` }];
  for (;;) {
    const task = await promptChat(screen, 'New task conversation', transcript);
    if (task == null) return;
    const request = /\.(mjs|json|md|markdown)$/.test(task) && !path.isAbsolute(task) ? path.resolve(currentWorkingDirectory, task) : task;
    try {
      const input = /\.(mjs|json|md|markdown)$/.test(request) ? '' : task;
      const job = startDetachedTasks([{ request, input, currentWorkingDirectory }])[0];
      transcript.push({ role: 'user', text: task });
      transcript.push({ role: 'pworker', text: `Task ${job.id.slice(0, 8)} started (${job.status}). Add another request or press Esc to return. Open Current tasks to monitor phases and results.` });
    } catch (error) {
      transcript.push({ role: 'user', text: task });
      transcript.push({ role: 'pworker', text: `Could not start the task: ${error.message}` });
    }
  }
}

async function currentTasks(screen, currentWorkingDirectory) {
  const store = jobStore(home);
  for (;;) {
    const jobs = store.list();
    const visible = jobs.slice(0, 10);
    const active = jobs.filter((job) => ['queued', 'compiling', 'running','waiting'].includes(job.status)).length;
    const selected = await selectMenu(screen, `Current tasks · ${active} active · newest ${visible.length}/${jobs.length}\nWorking directory: ${currentWorkingDirectory}`, [
      { label: 'Refresh task list', value: 'refresh' },
      ...visible.map((job) => ({ label: taskLabel(job), value: `task:${job.id}` })),
    ], { searchable: visible.length > 6 });
    if (!selected) return;
    if (selected === 'refresh') continue;
    await inspectTask(screen, selected.slice('task:'.length));
  }
}

async function interactive(currentWorkingDirectory) {
  await startProxy();
  const rl = !stdin.isTTY || !stdout.isTTY ? readline.createInterface({ input: stdin, output: stdout }) : null;
  const screen = ui(rl);
  try {
    for (;;) {
      const available = connectedProviders(loadLayers().config, await localStates());
      const choice = await selectMenu(screen, `Ploinky Workers · ${available.length} connected provider${available.length === 1 ? '' : 's'}\nWork folder: ${currentWorkingDirectory}`, [
        { label: 'Log in / connect provider', value: 'connect' },
        ...(available.length ? [{ label: 'Configure tier', value: 'tier' }] : []),
        ...(available.length ? [{ label: 'Auto-configure price ladder', value: 'ladder' }] : []),
        { label: 'New task conversation', value: 'compose' },
        { label: 'Current tasks', value: 'tasks' },
        { label: 'Manage local models', value: 'local' },
        { label: 'Statistics', value: 'stats' },
        { label: 'Stop proxy', value: 'stop' },
        { label: 'Exit', value: 'exit' },
      ]);
      if (!choice || choice === 'exit') return;
      try {
        if (choice === 'stop') { note(await stopMessage()); return; }
        if (choice === 'stats') { printStats({ since: '24h' }); continue; }
        if (choice === 'connect') {
          const connected = await connectProvider(screen);
          if (connected && await selectMenu(screen, `${connected.name} is ready`, [{ label: 'Configure a tier now', value: 'tier' }]) === 'tier') await configureTier(screen, connected.name);
          continue;
        }
        if (choice === 'tier') { await configureTier(screen); continue; }
        if (choice === 'ladder') { await configurePriceLadder(screen); continue; }
        if (choice === 'local') { await manageLocalModels(screen); continue; }
        if (choice === 'compose') { await composeTasks(screen, currentWorkingDirectory); continue; }
        if (choice === 'tasks') { await currentTasks(screen, currentWorkingDirectory); continue; }
      } catch (error) { note(`Operation failed: ${error.message}`); }
    }
  } finally { rl?.close(); }
}

async function main() {
  if (command === 'help' || args.includes('--help') || args.includes('-h')) { process.stdout.write(HELP); return; }
  ensureUserHome();
  if (!command) return interactive(fs.realpathSync(process.cwd()));
  if (command === '--cwd' && args.length === 2) {
    const folder = path.resolve(args[1]);
    if (!fs.existsSync(folder) || !fs.statSync(folder).isDirectory()) throw new Error(`Working directory does not exist: ${folder}`);
    return interactive(fs.realpathSync(folder));
  }
  // A directory opens the menu there, unless its name is a subcommand (a folder named "stats" must not hijack `pworker stats`).
  if (args.length === 1 && (!KNOWN_COMMANDS.has(command) || command.includes('/')) && fs.existsSync(command) && fs.statSync(command).isDirectory()) return interactive(fs.realpathSync(command));
  if (command === '_execute-batch') return executeDetachedBatch(args[1]);
  if (command === '--status' || command === 'status') {
    const store = jobStore(home);
    const found = args[1] ? publicJob(store.view(args[1])) : store.list().map(publicJob);
    if (args[1] && !found) { json({ error: `Task ${args[1]} does not exist` }); process.exitCode = 2; return; }
    json(found); return;
  }
  if (command === 'serve') {
    const { serve } = await import('../lib/server.mjs');
    const proxy = await serve({ port: option('port'), host: option('host') });
    if (proxy?.already) process.exitCode = 3;
    if (proxy?.conflict) process.exitCode = 4;
    return;
  }
  if (command === 'start') { await startProxy(); console.log('Proxy started.'); return; }
  if (command === 'stop') { const message = await stopMessage(); console.log(message); if (message.includes('still running')) process.exitCode = 1; return; }
  if (command === 'stats' && !args.includes('--proxy')) { printStats(); return; }
  // The engine verbs of `models` (catalog, run, status, ...). `hf-token` only writes a file and never needs the proxy; the others start it
  // when none runs. `models start|stop NAME` is not an engine verb: it goes on to the local servers below.
  if (isEngineCommand(args)) {
    if (args[1] !== 'hf-token') await startProxy();
    const abort = new AbortController();
    process.once('SIGINT', () => abort.abort());
    process.exitCode = await runEngineVerb({ args, client: client(), io: { out: (text) => stdout.write(text), err: (text) => process.stderr.write(text), stdin, stdout, isTTY: Boolean(stdout.isTTY), signal: abort.signal, home } });
    return;
  }
  if (['run', '--async', 'queue', 'flush', 'stats', 'models', 'tier'].includes(command)) await startProxy();
  if (command === 'stats') { json(await client().stats()); return; }
  if (command === 'models') {
    const action = args[1], provider = option('provider');
    if (provider) json(await modelsFromProxy(provider));
    else if (args.includes('--all')) {
      const available = connectedProviders(loadLayers().config, await localStates());
      const catalogs = await tierModelChoices(available);
      json({ models: catalogs.catalogues.flatMap((entry) => entry.models), errors: catalogs.unavailable });
      if (catalogs.unavailable.length) process.exitCode = 1;
    } else json(action === 'start' || action === 'stop' ? await client().model(args[2], action) : await client().models());
    return;
  }
  if (command === 'provider') {
    const name = args[1], key = option('key', '');
    const prior = loadLayers().config.providers?.[name] ?? {};
    if (prior.start) throw new Error('Use the local model menu to start a managed local provider');
    const spec = providerSpec(name, option('endpoint'), key, option('rpm', 60), prior);
    const models = await fetchModelCatalog(spec, { key });
    saveProvider(name, spec, key); await restartProxy();
    json({ provider: name, models: models.length }); return;
  }
  if (command === 'tier') {
    const name = option('provider'), model = option('model'), tierName = args[1];
    if (!tierName || tierName.startsWith('-') || !/^[A-Za-z][\w-]{0,39}$/.test(tierName) || tierName === 'maxWaitMs') throw new Error('Usage: pworker tier TIER --provider NAME --model MODEL_ID [--add] [--batch] (TIER: letters, digits, - and _)');
    if (!model) throw new Error('Missing --model MODEL_ID');
    const available = connectedProviders(loadLayers().config, await localStates());
    if (!available.includes(name)) throw new Error('Connect the provider before configuring a tier');
    const selected = (await modelsFromProxy(name)).find((entry) => entry.id === model);
    if (!selected) throw new Error(`Model ${model} is not in the live catalog of ${name}; nothing was saved`);
    if (!isModelEligibleForTier(selected)) throw new Error(`Model ${model} cannot serve a text tier: ${modelExclusionReason(selected)}`);
    const config = loadLayers().config;
    const existing = args.includes('--add') ? tierEntries(config, args[1]) : [];
    if (existing.some((entry) => entry.provider === name && entry.model === model)) throw new Error('That provider and model are already in the tier');
    const entries = [...existing, { provider: name, model }];
    saveModelLimits(selected);
    setTier(args[1], entries, args.includes('--batch') ? true : undefined); await restartProxy();
    json({ tier: args[1], entries }); return;
  }
  if (command === 'queue' && ['list', 'clear'].includes(args[1])) {
    if (args[1] === 'clear') { await withQueueLock(() => writeQueue([])); console.log('The queue is empty.'); }
    else json(readQueue());
    return;
  }
  if (command === 'run' || command === '--async' || command === 'queue') {
    const request = args[1] === '-' ? fs.readFileSync(0, 'utf8') : args[1];
    if (!request) throw new Error('Usage: pworker run <file.json|file.md|request|-> [--input text|JSON]');
    let input = option('input', ''); try { input = JSON.parse(input); } catch {}
    const currentWorkingDirectory = option('cwd', option('current-working-directory', null));
    if (command === 'queue') {
      // Paths are made absolute now: a later flush may run from another folder.
      if (isTaskFile(request) && !fs.existsSync(request)) throw new Error(`Task file does not exist: ${path.resolve(request)}`);
      const cwd = currentWorkingDirectory ? fs.realpathSync(path.resolve(currentWorkingDirectory)) : null;
      if (cwd && !fs.statSync(cwd).isDirectory()) throw new Error(`Working directory is not a directory: ${cwd}`);
      const count = await appendQueue([{ request: isTaskFile(request) ? path.resolve(request) : request, input, currentWorkingDirectory: cwd }]);
      console.log(`${count} tasks in the queue.`); return;
    }
    if (command === '--async' || args.includes('--async')) { json(startDetachedTasks([{ request, input, currentWorkingDirectory }])[0]); return; }
    const worker = new Pworker({ client: client(), config: loadLayers().config });
    const entry = await worker.enqueueRequest(request, input, { currentWorkingDirectory });
    const results = await worker.flush();
    json({ ...entry, results });
    if (results.some((r) => !r.ok)) process.exitCode = 1;
    return;
  }
  if (command === 'flush') {
    const { entries: pending, snapshot } = await takeQueue();
    const finish = async (failed) => { if (failed.length) await appendQueue(failed); if (snapshot) fs.rmSync(snapshot, { force: true }); };
    if (args.includes('--async')) {
      let jobs = [];
      try { jobs = pending.length ? startDetachedTasks(pending) : []; }
      catch (error) { await finish(pending); throw error; }
      await finish([]);
      json(jobs); return;
    }
    const worker = new Pworker({ client: client(), config: loadLayers().config });
    const entries = [], failed = [], byId = new Map();
    // One bad entry (a missing file, an invalid task) fails alone and stays queued; the others run.
    for (const item of pending) {
      try { const entry = await worker.enqueueRequest(item.request, item.input, { currentWorkingDirectory: item.currentWorkingDirectory }); entries.push(entry); byId.set(entry.id, item); }
      catch (error) { entries.push({ request: item.request, error: error.message }); failed.push(item); }
    }
    let results = [];
    try { results = await worker.flush(); }
    finally { await finish([...failed, ...results.filter((r) => !r.ok).map((r) => byId.get(r.id)).filter(Boolean), ...(results.length ? [] : [...byId.values()])]); }
    json({ entries, results });
    if (failed.length || results.some((r) => !r.ok)) process.exitCode = 1;
    return;
  }
  console.log('pworker [run <task.json|task.md|request|-> --input text|JSON [--cwd DIR] [--async]] | --status [TASK_ID] | queue <task> --input ... [--cwd DIR] | queue list|clear | flush [--async] | start | stop | serve | stats [--since 1h|24h|7d] [--provider P] [--by hour|day] [--json] [--proxy] | models [start|stop <local>|catalog|runners|run|stop|status|...] | provider <name> --endpoint URL --key KEY --rpm N | tier <name> --provider P --model ID [--batch]');
}
main().catch((e) => { console.error(e.message); process.exitCode = 1; });
